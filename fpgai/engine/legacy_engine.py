
import onnx
from fpgai.frontend.onnx.reader import get_model_weights, get_model_arch, verify_model

from utils.main_function_handler import construct_main_function
from utils.generate_object_rep import generate_obj_rep
from utils.get_nodes import get_number_of_input_nodes, get_number_of_output_nodes
from utils.get_pragma import get_pragmas
from utils.write_cpp_file import write_cpp_file, write_header_file, write_tcl_file
from utils.read_stream import read_input_stream
from utils.write_stream import write_output_stream
from utils.add_linear_act import add_linear_activation
from utils.vitis_tcl import vitis_tcl_generator
from utils.vivado_tcl import vivado_tcl_generator
from utils.testbench import generate_testbench_codes

import os
import numpy as np

from fpgai.runtime.onnxruntime_backend import run_onnxruntime


from loss_functions.loss import *


class fpgai_engine:
    """
    FPGAI Engine:
      - Loads ONNX
      - Verifies model
      - Builds architecture representation
      - Generates HLS C++ / headers / TCL scripts / testbench
      - (Optionally) runs reference inference/training via onnxruntime helpers
    """

    def __init__(
        self,
        input_data,
        first_layer_shape,
        output_shape,
        learning_rate: float = 0.1,
        mode: str = "inference",
        onnx_file_name: str = "mlp.onnx",
        precision: str = "float",
        verbose: bool = False,   # NEW: gate debug prints
    ):
        # -----------------------------
        # Core settings
        # -----------------------------
        self.verbose = verbose
        self.main_func_name = "deeplearn"
        self.precision = precision
        self.loss_function = MeanSquaredError()
        self.learning_rate = learning_rate
        self.mode = mode
        self.onnx_file_name = onnx_file_name

        # -----------------------------
        # Load ONNX + verify
        # -----------------------------
        self.obj_arch_rep = []
        self.model = onnx.load(self.onnx_file_name)

        verify_str = self.verify_onnx_model()
        if verify_str != 1:
            raise Exception(verify_str)

        # -----------------------------
        # Extract model structure + weights
        # -----------------------------
        self.layers = self.get_model_arch()
        self.strides = []
        self.kernels = []
        self.weights = self.get_weights()
        self.layer_function_implementations = ""
        self.first_layer_shape = first_layer_shape

        # -----------------------------
        # Build internal representation
        # -----------------------------
        add_linear_activation(self)
        generate_obj_rep(self)

        if self.verbose:
            print(self.layers)

        self.number_of_input_nodes = get_number_of_input_nodes(self)
        self.number_of_output_nodes = get_number_of_output_nodes(self)

        # -----------------------------
        # Code generation
        # -----------------------------
        generated_hls_code = self.generate_hls_codes()
        generated_header_code = self.generate_header_codes()
        generated_vitis_tcl = self.generate_vitis_tcl_codes()
        generated_vivado_tcl = self.generate_vivado_tcl_codes()

        # -----------------------------
        # Reference runtime (host-side)
        # -----------------------------
        # IMPORTANT FIX: use self.onnx_file_name (not the free variable)
        # Also pass verbose down (requires onnx_inference_pytorch to accept verbose; see note below)
        try:
            run_onnxruntime(self.onnx_file_name, input_data, verbose=self.verbose)

        except TypeError:
            # Backward compatibility if onnx_inference_pytorch doesn't accept verbose yet
            run_onnxruntime(self.onnx_file_name, input_data, verbose=self.verbose)


        # -----------------------------
        # Testbench data shaping
        # -----------------------------
        # Normalize output_shape to a 4D shape the testbench expects
        out_shape = tuple(output_shape)  # e.g., (1,10) or (1,1,1,4)
        if len(out_shape) == 2:
            # (batch, features) -> (1,1,1,features)
            _, feats = out_shape
            out_shape = (1, 1, 1, int(feats))
        elif len(out_shape) == 3:
            # pad to 4D
            out_shape = (1,) + tuple(int(d) for d in out_shape)
        elif len(out_shape) > 4:
            # collapse anything >4D to (1,1,1,total)
            total = int(np.prod(out_shape))
            out_shape = (1, 1, 1, total)
        else:
            # ensure ints and no zeros/Nones
            out_shape = tuple(1 if (d is None or d == 0) else int(d) for d in out_shape)

        increment = 0.1
        total_elements = int(np.prod(out_shape))
        target_output = (np.arange(total_elements, dtype=np.float32) + 1) * increment
        target_output = target_output.reshape(out_shape)

        target_output = target_output.flatten()
        input_data_flat = input_data.flatten()

        if self.mode == "inference":
            target_output = None

        # -----------------------------
        # Generate testbench code
        # -----------------------------
        generated_testbench_code = generate_testbench_codes(
            input_data=input_data_flat,
            output_file_dest=os.getcwd() + "/generated_files",
            target_output=target_output,
            model=self,
        )

        # -----------------------------
        # Write artifacts
        # -----------------------------
        write_cpp_file("generated_files/main", generated_hls_code)
        write_cpp_file("generated_files/testbench", generated_testbench_code)
        write_header_file("generated_files/deeplearn", generated_header_code)
        write_tcl_file("generated_files/tcl_for_vitis", generated_vitis_tcl)
        write_tcl_file("generated_files/tcl_for_vivado", generated_vivado_tcl)

    # -------------------------------------------------------------------------
    # Model helpers
    # -------------------------------------------------------------------------
    def verify_onnx_model(self):
        return verify_model(self.model)

    def get_model_arch(self):
        return get_model_arch(self.model)

    def get_weights(self):
        return get_model_weights(self.model)

    # -------------------------------------------------------------------------
    # Codegen
    # -------------------------------------------------------------------------
    def generate_hls_codes(self):
        generated_hls_codes = ""

        # layer code
        for i in range(len(self.obj_arch_rep)):
            generated_hls_codes += self.obj_arch_rep[i].get_hls_file_string()

        generated_hls_codes += self.layer_function_implementations
        generated_hls_codes += construct_main_function(self)
        generated_hls_codes += get_pragmas(self)
        generated_hls_codes += read_input_stream(self)

        if self.mode == "inference":
            for i in range(len(self.obj_arch_rep)):
                generated_hls_codes += self.obj_arch_rep[i].get_inference_func()

        if self.mode == "training":
            for i in range(len(self.obj_arch_rep)):
                generated_hls_codes += self.obj_arch_rep[i].get_inference_func()
            for i in range(len(self.obj_arch_rep) - 1, -1, -1):
                generated_hls_codes += self.obj_arch_rep[i].get_delta_calculation_func()
            for i in range(len(self.obj_arch_rep) - 1, -1, -1):
                generated_hls_codes += self.obj_arch_rep[i].get_update_weights_func()

        generated_hls_codes += write_output_stream(self)
        return generated_hls_codes

    def generate_header_codes(self):
        generated_header_code = ""
        for i in range(len(self.obj_arch_rep)):
            generated_header_code += self.obj_arch_rep[i].get_header_file_string()
        return generated_header_code

    def generate_vitis_tcl_codes(self):
        return vitis_tcl_generator(
            src_file_dir=os.getcwd() + "/generated_files",
            project_name="compiler_hls_project",
            project_dir=os.getcwd() + "/generated_files/",
        )

    def generate_vivado_tcl_codes(self):
        return vivado_tcl_generator(
            project_name="compiler_vivado_project",
            project_dir=os.getcwd() + "generated_files/",
            ip_repo_dir=os.getcwd() + "generated_files/compiler_hls_project",
        )
