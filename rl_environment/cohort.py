"""Deterministic frozen functional networks; independent topology cohorts.

Models are true FPGAI Functional Model IRs, not synthetic graph tensors.
Currently 1–4 MatMul/Gemm only, matching the real HLS architecture catalog.
"""
from __future__ import annotations
from random import Random
import numpy as np

from fpgai_research.model_ir.contracts import Tensor, Operator, FunctionalModel
from .tasks import Task, DEFAULT_PART


def network(widths, kinds, *, seed):
    widths=tuple(widths)
    if (not 2<=len(widths)<=5 or any(type(x) is not int or not 2<=x<=128 for x in widths)
            or len(kinds)!=len(widths)-1 or any(x not in ("MatMul","Gemm") for x in kinds)):
        raise ValueError("Only bounded 1..4 layer MatMul/Gemm functional models supported")
    rng=np.random.default_rng(seed)
    tensors=[Tensor("input",(1,widths[0]))]
    operators=[]
    last="input"
    for i,(left,right,kind) in enumerate(zip(widths,widths[1:],kinds)):
        w=f"weight_{i}"
        y=f"layer_{i}"
        values=tuple(float(x) for x in rng.uniform(-.3,.3,size=left*right))
        tensors.extend((Tensor(w,(left,right),data=values),Tensor(y,(1,right))))
        operators.append(Operator(f"op_{i}",kind,(last,w),(y,)))
        last=y
    model=FunctionalModel(tuple(tensors),tuple(operators),("input",),(last,))
    model.validate()
    return model


def make_cohort(*,training=100,validation=6,test=8,seed=20261030,
                max_vitis_calls=3,part=DEFAULT_PART,clock_ns=10.0,
                legal_clocks_ns=(), fixed_clock_ablation_ns=None,
                max_absolute_error=.05,
                resources=(93696,187392,998,230,51),
                freeze_numeric_format=False, beginner_pilot=False,
                curriculum_stage=None):
    if any(type(n) is not int or not 1<=n<=10000
           for n in (training,validation,test)):
        raise ValueError("Require 1..10000 networks per cohort")
    if type(seed) is not int or not 0<=seed<2**32-40000:
        raise ValueError("Invalid cohort seed")
    if type(beginner_pilot) is not bool:
        raise ValueError("beginner_pilot must be explicitly boolean")
    from .curriculum import get_stage, stage_validation
    stage = get_stage(curriculum_stage)
    if stage is not None:
        if clock_ns is not None:
            raise ValueError("Curriculum source tasks must use policy or fixed-ablation clock")
        stage_validation(
            stage, fixed_clock=fixed_clock_ablation_ns,
            frozen_numeric=freeze_numeric_format, beginner=beginner_pilot)
    if beginner_pilot and (clock_ns is not None
                           or fixed_clock_ablation_ns != 32.
                           or not freeze_numeric_format):
        raise ValueError("Beginner cohort requires 32 ns fixed clock, 16/6 precision")
    # Pre-registered, genuinely distinct width families: the test has never
    # seen a layer of these (input,output) dimensions in training.
    beginner_shapes = {
        "train": ((4, 4), (4, 8), (8, 4), (8, 8)),
        "validation": ((4, 6), (6, 4)),
        "test": ((6, 8), (8, 6)),
    }
    stage_shapes = {
        "L1": {
            "train": ((4,4),(4,8),(8,4),(8,8),(6,6)),
            "validation": ((4,6),(6,4)),
            "test": ((6,8),(8,6)),
        },
        "L2": {
            "train": ((4,4,4),(4,8,4),(8,4,8),(8,8,8)),
            "validation": ((4,6,4),(6,4,6)),
            "test": ((6,8,6),(8,6,8)),
        },
    }
    rng=Random(seed)
    result={"train":[],"validation":[],"test":[]}
    index=0
    for partition,count in (("train",training),("validation",validation),("test",test)):
        for j in range(count):
            if beginner_pilot:
                widths = beginner_shapes[partition][j % len(beginner_shapes[partition])]
                depth = 1
                kinds = ("MatMul",)
            elif stage is not None and stage.name in stage_shapes:
                widths = stage_shapes[stage.name][partition][
                    j % len(stage_shapes[stage.name][partition])]
                depth = len(widths) - 1
                kinds = ("MatMul",) * depth if stage.name == "L1" else (
                    ("MatMul", "Gemm") if j % 2 else ("MatMul", "MatMul"))
            elif stage is not None and stage.name in ("L3", "L4", "L5"):
                depth = 1 + j % 2
                widths = tuple(rng.randint(3, 10) for _ in range(depth + 1))
                kinds = tuple(
                    "Gemm" if (j >> axis) & 1 else "MatMul"
                    for axis in range(depth))
            else:
                depth=1+j%2 if partition=="train" else (
                    3 if partition=="validation" else 4)
                widths=tuple(rng.randint(3,10) for _ in range(depth+1))
                kinds=tuple("Gemm" if (j>>axis)&1 else "MatMul"
                            for axis in range(depth))
            model=network(widths,kinds,seed=seed+index+1)
            values=np.random.default_rng(seed+50000+index).uniform(
                -.25,.25,size=(3,1,widths[0])).astype(np.float32)
            result[partition].append(Task(
                task_id=f"{partition}_{j:05d}",
                model=model,samples=values,part=part,clock_ns=clock_ns,
                max_absolute_error=max_absolute_error,
                resources=resources,max_vitis_calls=max_vitis_calls,
                legal_clocks_ns=legal_clocks_ns,
                fixed_clock_ablation_ns=fixed_clock_ablation_ns,
                freeze_numeric_format=freeze_numeric_format,
                beginner_pilot=beginner_pilot,
                curriculum_stage=curriculum_stage,
                partition=partition))
            index+=1
    # Holdout by operator topology count; no train models in heldout sets.
    expected_depths = (
        ((("train",{1}),("validation",{1}),("test",{1}))
         if beginner_pilot else
         (("train",{1,2}),("validation",{3}),("test",{4})))
        if stage is None else
        (("train",set(stage.train_depths)),
         ("validation",set(stage.val_depths)),
         ("test",set(stage.test_depths))))
    for partition, depths in expected_depths:
        if any(len(t.model.operators) not in depths for t in result[partition]):
            raise AssertionError("Pre-registered topology cohort failed")
    if beginner_pilot:
        for partition in ("train", "validation", "test"):
            if any(t.model.operators[0].kind != "MatMul" for t in result[partition]):
                raise AssertionError("Beginner pilot contains non-MatMul operators")
        def shapes(split):
            return {
                (t.model.tensor(t.model.inputs[0]).shape[-1],
                 t.model.tensor(t.model.outputs[0]).shape[-1])
                for t in result[split]
            }
        if shapes("train") & (shapes("validation") | shapes("test")):
            raise AssertionError("Beginner heldout dimensions leak into training")
    if len({t.model_sha for tasks in result.values() for t in tasks})!=index:
        raise ValueError("Functional model identity leaked across splits")
    return result
