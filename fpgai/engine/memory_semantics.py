from __future__ import annotations

from typing import Any, Dict

from fpgai.backends.vivado.boards import get_board
from fpgai.config.access import get_path


_cfg_get = get_path


class MemorySemanticsMixin:
    """Resolve weight, activation, and gradient memory semantics.

    The mixin keeps the compiler orchestration small while preserving the
    existing ``Compiler`` method API used by tests and internal callers.
    """

    @staticmethod
    def _normalise_weight_storage(value: Any) -> str:
        raw_value = str(value or "").strip().lower().replace("-", "_")
        aliases = {
            "embedded": "bram",
            "on_chip": "bram",
            "onchip": "bram",
            "block": "bram",
            "block_ram": "bram",
            "bram": "bram",
            "uram": "uram",
            "ultra": "uram",
            "ultra_ram": "uram",
            "ddr": "ddr",
            "external": "ddr",
            "external_ddr": "ddr",
            "dma_ddr": "ddr",
        }
        return aliases.get(raw_value, raw_value)

    @staticmethod
    def _normalise_movement_interface(value: Any) -> str:
        raw_value = str(value or "").strip().lower().replace("-", "_")
        aliases = {
            "compile": "compile_time",
            "compiletime": "compile_time",
            "compiled": "compile_time",
            "static": "compile_time",
            "embedded": "compile_time",
            "const": "compile_time",
            "bram": "compile_time",
            "uram": "compile_time",
            "axi_dma": "axi_stream",
            "axis": "axi_stream",
            "stream": "axi_stream",
            "streamed": "axi_stream",
            "dma": "axi_stream",
            "ddr": "m_axi",
            "dma_ddr": "m_axi",
            "external": "m_axi",
            "external_ddr": "m_axi",
            "m_axi": "m_axi",
            "maxi": "m_axi",
            "none": "none",
            "off": "none",
        }
        return aliases.get(raw_value, raw_value)

    @staticmethod
    def _normalise_movement_policy(value: Any) -> str:
        raw_value = str(value or "").strip().lower().replace("-", "_")
        aliases = {
            "static": "static",
            "compile_time": "static",
            "compiletime": "static",
            "full": "full",
            "preload": "full",
            "preload_full": "full",
            "import_full": "full",
            "load_full": "full",
            "tiled": "tiled",
            "tile": "tiled",
            "stream": "tiled",
            "streaming": "tiled",
            "none": "none",
            "off": "none",
        }
        return aliases.get(raw_value, raw_value)

    @staticmethod
    def _normalise_transport(value: Any, interface: str) -> str:
        raw_value = str(value or "").strip().lower().replace("-", "_")
        aliases = {
            "axi_dma": "dma",
            "dma": "dma",
            "ps": "ps_runtime",
            "ps_ddr": "ps_runtime",
            "runtime": "ps_runtime",
            "host": "ps_runtime",
            "none": "none",
            "off": "none",
        }
        if raw_value:
            return aliases.get(raw_value, raw_value)
        if interface == "axi_stream":
            return "dma"
        if interface == "m_axi":
            return "ps_runtime"
        return "none"

    @staticmethod
    def _has_explicit_weight_data_movement(raw: Dict[str, Any]) -> bool:
        """Return True when the user provided detailed weight import/export movement.

        weights.mode is an intent shortcut.  Detailed data_movement entries remain
        higher priority and must not be overwritten by the shortcut expansion.
        Legacy data_movement.ps_pl.weights.mode is treated as a legacy/default
        shortcut, so an explicit top-level weights.mode may override it.
        """
        explicit_paths = (
            "data_movement.weights.import",
            "data_movement.weights.load",
            "data_movement.weights.export",
            "data_movement.weights.store",
        )
        for path in explicit_paths:
            value = _cfg_get(raw, path, None)
            if isinstance(value, dict) and value:
                return True
        scalar_paths = (
            "data_movement.weights.import.interface",
            "data_movement.weights.load.interface",
            "data_movement.weights.export.interface",
            "data_movement.weights.store.interface",
        )
        return any(_cfg_get(raw, path, None) is not None for path in scalar_paths)

    @staticmethod
    def _normalise_user_weight_mode(value: Any) -> str:
        mode = str(value or "").strip().lower().replace("-", "_")
        aliases = {
            "": "",
            "static": "embedded",
            "compile_time": "embedded",
            "compiletime": "embedded",
            "const": "embedded",
            "on_chip": "embedded",
            "onchip": "embedded",
            "embedded": "embedded",
            "runtime_import": "import",
            "runtime_loaded": "import",
            "load": "import",
            "import": "import",
            "load_store": "import_export",
            "import_store": "import_export",
            "import_export": "import_export",
            "export_import": "import_export",
            "ddr": "tiled",
            "ddr_tiled": "tiled",
            "tile": "tiled",
            "tiled": "tiled",
            "mutable_tiled": "tiled_mutable",
            "ddr_tiled_mutable": "tiled_mutable",
            "tiled_mutable": "tiled_mutable",
        }
        return aliases.get(mode, mode)

    def _expand_user_weight_mode_to_movement(
        self,
        raw: Dict[str, Any],
        *,
        storage: str,
    ) -> tuple[Dict[str, Any], Dict[str, Any], str | None]:
        """Expand weights.mode / training.weight_initialization.mode to import/export cfg.

        Returns (import_cfg, export_cfg, source_mode).  source_mode is the
        normalized high-level mode that produced the expansion, or None when no
        user-facing shortcut was present.
        """
        mode_value = _cfg_get(raw, "weights.mode", None)
        source_path = "weights.mode"
        if mode_value is None:
            init_mode = _cfg_get(raw, "training.weight_initialization.mode", None)
            if init_mode is not None:
                init = str(init_mode or "").strip().lower().replace("-", "_")
                init_aliases = {
                    "compile_time": "embedded",
                    "compiletime": "embedded",
                    "static": "embedded",
                    "embedded": "embedded",
                    "const": "embedded",
                    "import": "import",
                    "runtime_import": "import",
                    "load": "import",
                }
                if init in {"zero", "xavier", "he", "random", "random_seeded", "seeded_random"}:
                    raise ValueError(
                        "training.weight_initialization.mode={!r} is not implemented yet. "
                        "Supported modes are compile_time and import.".format(init_mode)
                    )
                mode_value = init_aliases.get(init, init)
                source_path = "training.weight_initialization.mode"
        mode = self._normalise_user_weight_mode(mode_value)
        if not mode:
            return {}, {}, None

        pipeline_mode = str(_cfg_get(raw, "pipeline.mode", "inference") or "inference").strip().lower()
        if mode == "embedded":
            if storage == "ddr":
                raise ValueError(
                    f"{source_path}=embedded is only valid with BRAM/URAM weight storage. "
                    "DDR storage means tiled external-memory execution; use weights.mode=tiled."
                )
            return (
                {"interface": "compile_time", "transport": "none", "policy": "static"},
                {"interface": "none", "transport": "none", "policy": "none"},
                mode,
            )
        if mode == "import":
            if storage == "ddr":
                raise ValueError(
                    f"{source_path}=import is only valid with BRAM/URAM weight storage. "
                    "Full import into local memory is not DDR storage; use BRAM/URAM or weights.mode=tiled."
                )
            return (
                {"interface": "m_axi", "transport": "ps_runtime", "policy": "full"},
                {"interface": "none", "transport": "none", "policy": "none"},
                mode,
            )
        if mode == "import_export":
            if storage == "ddr":
                raise ValueError(
                    f"{source_path}=import_export is only valid with BRAM/URAM weight storage. "
                    "DDR mutable weights must use weights.mode=tiled_mutable."
                )
            return (
                {"interface": "m_axi", "transport": "ps_runtime", "policy": "full"},
                {"interface": "m_axi", "transport": "ps_runtime", "policy": "full"},
                mode,
            )
        if mode == "tiled":
            if storage != "ddr":
                raise ValueError(
                    f"{source_path}=tiled requires memory.storage.weights=ddr. "
                    f"Got memory.storage.weights={storage!r}."
                )
            return (
                {"interface": "m_axi", "transport": "ps_runtime", "policy": "tiled"},
                {"interface": "none", "transport": "none", "policy": "none"},
                mode,
            )
        if mode == "tiled_mutable":
            if storage != "ddr":
                raise ValueError(
                    f"{source_path}=tiled_mutable requires memory.storage.weights=ddr. "
                    f"Got memory.storage.weights={storage!r}."
                )
            if pipeline_mode != "training_on_device":
                raise ValueError(
                    f"{source_path}=tiled_mutable is a training weight mode. "
                    f"Got pipeline.mode={pipeline_mode!r}."
                )
            return (
                {"interface": "m_axi", "transport": "ps_runtime", "policy": "tiled"},
                {"interface": "m_axi", "transport": "ps_runtime", "policy": "tiled"},
                mode,
            )
        raise ValueError(
            f"Unsupported {source_path}={mode_value!r}. Supported values are "
            "embedded, import, import_export, tiled, and tiled_mutable."
        )

    def _reject_unsupported_training_weight_storage(self, raw: Dict[str, Any]) -> None:
        pipeline_mode = str(_cfg_get(raw, "pipeline.mode", "") or "").strip().lower()
        if pipeline_mode != "training_on_device":
            return
        training_weight_storage = str(
            _cfg_get(
                raw,
                "training.storage.weights",
                _cfg_get(raw, "memory.storage.weights", _cfg_get(raw, "memory.weight_storage", "bram")),
            )
            or "bram"
        ).strip().lower().replace("-", "_")
        weight_storage = str(
            _cfg_get(raw, "memory.storage.weights", _cfg_get(raw, "memory.weight_storage", training_weight_storage))
            or training_weight_storage
        ).strip().lower().replace("-", "_")
        aliases = {"ultra": "uram", "ultra_ram": "uram", "external": "ddr", "external_ddr": "ddr", "dma_ddr": "ddr"}
        training_weight_storage = aliases.get(training_weight_storage, training_weight_storage)
        weight_storage = aliases.get(weight_storage, weight_storage)
        # DDR training now maps to the tiled mutable backend for Dense/Conv graphs.
        # Unsupported layer types are rejected by top_train_cpp so the compiler
        # no longer rejects the storage choice before codegen.
        if training_weight_storage == "uram" or weight_storage == "uram":
            board_name = str(_cfg_get(raw, "targets.board", _cfg_get(raw, "project.board", "")) or "").strip()
            if board_name:
                try:
                    board = get_board(board_name)
                    if int(board.uram or 0) <= 0:
                        raise ValueError(
                            f"memory.storage.weights=uram requires URAM, but selected board {board_name!r} has 0 URAM blocks."
                        )
                except KeyError:
                    pass

    def _resolve_weight_movement_semantics(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        requested_storage = _cfg_get(
            raw,
            "memory.storage.weights",
            _cfg_get(raw, "memory.weight_storage", None),
        )

        # Legacy configs sometimes specified only data_movement.ps_pl.weights.mode.
        # Treat that as a legacy/default shortcut, not as detailed movement;
        # top-level weights.mode is the newer user-facing intent and should win.
        legacy_mode = None if _cfg_get(raw, "weights.mode", None) is not None else _cfg_get(raw, "data_movement.ps_pl.weights.mode", None)
        storage = self._normalise_weight_storage(requested_storage or "bram")

        explicit_weight_movement = self._has_explicit_weight_data_movement(raw)
        expanded_user_mode: str | None = None
        if explicit_weight_movement:
            import_cfg = _cfg_get(raw, "data_movement.weights.import", None)
            if not isinstance(import_cfg, dict):
                import_cfg = _cfg_get(raw, "data_movement.weights.load", None)
            if not isinstance(import_cfg, dict):
                import_cfg = {}
            export_cfg = _cfg_get(raw, "data_movement.weights.export", None)
            if not isinstance(export_cfg, dict):
                export_cfg = _cfg_get(raw, "data_movement.weights.store", None)
            if not isinstance(export_cfg, dict):
                export_cfg = {}
        else:
            import_cfg, export_cfg, expanded_user_mode = self._expand_user_weight_mode_to_movement(
                raw,
                storage=storage,
            )

        legacy_interface = _cfg_get(raw, "data_movement.weights.load.interface", legacy_mode)
        import_interface = self._normalise_movement_interface(
            import_cfg.get("interface", legacy_interface)
            or ("m_axi" if storage == "ddr" else "compile_time")
        )
        import_policy = self._normalise_movement_policy(
            import_cfg.get("policy", None)
            or ("tiled" if storage == "ddr" else "static")
        )
        import_transport = self._normalise_transport(import_cfg.get("transport", None), import_interface)

        export_interface = self._normalise_movement_interface(export_cfg.get("interface", "none"))
        export_policy = self._normalise_movement_policy(export_cfg.get("policy", "none"))
        export_transport = self._normalise_transport(export_cfg.get("transport", None), export_interface)

        if storage not in {"bram", "uram", "ddr"}:
            raise ValueError(
                "memory.storage.weights must be one of bram, uram, or ddr; "
                f"got {requested_storage!r}. Use data_movement.weights.import/export "
                "for transfer behavior."
            )

        if storage == "bram":
            if import_interface == "compile_time" and import_policy == "static":
                resolved = "bram_static"
                hls_mode = "embedded"
                payload_required = False
            elif import_interface == "m_axi" and import_policy == "full":
                resolved = "bram_import_export_full" if export_interface == "m_axi" and export_policy == "full" else "bram_import_full"
                hls_mode = "ddr"
                payload_required = True
            else:
                raise ValueError(
                    "BRAM weight storage supports import compile_time/static or m_axi/full. "
                    f"Got import {import_interface}/{import_policy}."
                )
        elif storage == "uram":
            if import_interface == "compile_time" and import_policy == "static":
                resolved = "uram_static"
                hls_mode = "embedded"
                payload_required = False
            elif import_interface == "m_axi" and import_policy == "full":
                resolved = "uram_import_export_full" if export_interface == "m_axi" and export_policy == "full" else "uram_import_full"
                hls_mode = "uram"
                payload_required = True
            else:
                raise ValueError(
                    "URAM weight storage supports import compile_time/static or m_axi/full. "
                    f"Got import {import_interface}/{import_policy}."
                )
        else:
            if import_interface == "m_axi" and import_policy == "tiled":
                if export_interface not in {"none", "m_axi"} or (export_interface == "m_axi" and export_policy != "tiled"):
                    raise ValueError(
                        "DDR weight storage supports export none/none for inference or m_axi/tiled for future mutable training. "
                        f"Got export {export_interface}/{export_policy}."
                    )
                resolved = "ddr_tiled_mutable" if export_interface == "m_axi" and export_policy == "tiled" else "ddr_tiled"
                hls_mode = resolved
                payload_required = True
            else:
                raise ValueError(
                    "DDR weight storage means weights stay in DDR and must use m_axi/tiled import. "
                    f"Got import {import_interface}/{import_policy}. Full import belongs to BRAM/URAM storage."
                )

        if storage != "ddr" and export_interface != "none" and not (export_interface == "m_axi" and export_policy == "full"):
            raise ValueError(
                "Weight export currently supports none/none or m_axi/full only for BRAM/URAM storage; "
                f"got export {export_interface}/{export_policy}."
            )

        return {
            "requested_weight_storage": str(requested_storage or storage),
            "resolved_weight_storage": storage,
            "resolved_weight_semantics": resolved,
            "memory_semantics_mode": resolved,
            "hls_weights_mode": hls_mode,
            "runtime_weight_payload_required": payload_required,
            "full_local_weight_replica": storage in {"bram", "uram"},
            "tile_weight_buffer": storage == "ddr",
            "scalable_external_weight_execution": storage == "ddr",
            "weight_import_interface": import_interface,
            "weight_import_transport": import_transport,
            "weight_import_policy": import_policy,
            "weight_export_interface": export_interface,
            "weight_export_transport": export_transport,
            "weight_export_policy": export_policy,
            "user_weight_mode": expanded_user_mode,
            "weight_movement_source": "data_movement" if explicit_weight_movement else ("weights.mode" if expanded_user_mode else "compiler_default"),
            "runtime_commands_supported": [
                *(["import_weights"] if payload_required else []),
                "run_inference",
                *(["export_weights"] if export_interface == "m_axi" and export_policy == "full" else []),
            ],
            "reload_before_each_compute": False,
        }

    def _resolve_activation_storage_semantics(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        requested = _cfg_get(raw, "memory.storage.activations", _cfg_get(raw, "memory.activation_storage", "bram"))
        value = str(requested or "bram").strip().lower().replace("-", "_")
        aliases = {
            "block": "bram",
            "block_ram": "bram",
            "bram": "bram",
            "ultra": "uram",
            "ultra_ram": "uram",
            "uram": "uram",
        }
        storage = aliases.get(value)
        if storage not in {"bram", "uram"}:
            raise ValueError(
                "memory.storage.activations must be bram or uram for this backend; "
                f"got {requested!r}."
            )
        board_name = str(_cfg_get(raw, "targets.board", _cfg_get(raw, "project.board", "")) or "").strip()
        if storage == "uram" and board_name:
            try:
                board = get_board(board_name)
                if int(board.uram or 0) <= 0:
                    raise ValueError(
                        f"memory.storage.activations=uram requires URAM, but selected board {board_name!r} has 0 URAM blocks."
                    )
            except KeyError:
                # Unknown-board validation is handled by the board/backend path.
                pass
        return {
            "requested_activation_storage": str(requested or storage),
            "resolved_activation_storage": storage,
            "activation_storage_semantics": f"activation_{storage}",
            "activation_local_buffers": True,
        }

    def _resolve_hls_weights_mode(self, raw: Dict[str, Any]) -> str:
        """Resolve storage + import/export movement into the existing HLS backend mode."""
        return str(self._resolve_weight_movement_semantics(raw)["hls_weights_mode"])

    def _annotate_memory_movement_semantics(self, compile_plan, memory_plan, raw: Dict[str, Any]) -> Dict[str, Any]:
        semantics = self._resolve_weight_movement_semantics(raw)
        activation_semantics = self._resolve_activation_storage_semantics(raw)
        semantics.update(activation_semantics)
        gradient_computation = str(_cfg_get(raw, "training.gradients.computation", "full_buffer") or "full_buffer").strip().lower().replace("-", "_")
        if gradient_computation not in {"full_buffer", "tiled_accumulate", "fused_update"}:
            raise ValueError(
                "training.gradients.computation must be full_buffer, tiled_accumulate, or fused_update; "
                f"got {gradient_computation!r}."
            )

        requested_gradient_storage = _cfg_get(
            raw,
            "training.storage.parameter_gradient",
            _cfg_get(
                raw,
                "training.storage.gradient",
                _cfg_get(raw, "training.storage.gradients", _cfg_get(raw, "memory.storage.gradients", None)),
            ),
        )
        gradient_storage_was_explicit = requested_gradient_storage is not None
        default_gradient_storage = "recompute" if gradient_computation == "fused_update" else "bram"
        gradient_storage = str(requested_gradient_storage or default_gradient_storage).strip().lower().replace("-", "_")
        gradient_aliases = {"block": "bram", "block_ram": "bram", "bram": "bram", "ultra": "uram", "ultra_ram": "uram", "uram": "uram"}
        gradient_storage = gradient_aliases.get(gradient_storage, gradient_storage)
        if gradient_storage not in {"bram", "uram", "ddr", "recompute"}:
            raise ValueError(
                "training.storage.parameter_gradient must be bram, uram, ddr, or recompute; "
                f"got {gradient_storage!r}."
            )
        if gradient_computation == "fused_update" and gradient_storage != "recompute":
            raise ValueError(
                "training.gradients.computation=fused_update does not materialize a parameter-gradient buffer; "
                "omit training.storage.parameter_gradient or set it to recompute."
            )
        if gradient_computation == "tiled_accumulate":
            optimizer_type = str(_cfg_get(raw, "training.optimizer.type", "sgd") or "sgd").strip().lower().replace("-", "_")
            batch_mode = str(_cfg_get(raw, "training.batch.mode", "direct") or "direct").strip().lower().replace("-", "_")
            batch_size = int(_cfg_get(raw, "training.batch.size", 1) or 1)
            accumulation_steps = int(_cfg_get(raw, "training.gradient_accumulation.steps", 1) or 1)
            if optimizer_type != "adam":
                raise ValueError(
                    "training.gradients.computation=tiled_accumulate currently supports training.optimizer.type=adam."
                )
            if batch_mode in {"accumulated", "accumulate", "gradient_accumulation"} or batch_size != 1 or accumulation_steps != 1:
                raise ValueError(
                    "training.gradients.computation=tiled_accumulate currently requires direct single-record updates "
                    "(training.batch.mode=direct, training.batch.size=1, gradient_accumulation.steps=1)."
                )
        if gradient_storage == "ddr":
            raise ValueError(
                "training.storage.parameter_gradient=ddr requires a real external-memory lowering and is not yet enabled; "
                "use bram, uram, or select fused_update with recompute storage."
            )
        if gradient_storage == "recompute" and gradient_computation != "fused_update":
            raise ValueError(
                "training.storage.parameter_gradient=recompute currently requires "
                "training.gradients.computation=fused_update."
            )
        requested_gradient_materialization = _cfg_get(raw, "training.gradients.materialization", None)
        default_gradient_materialization = "streamed" if gradient_computation == "fused_update" else "full"
        gradient_materialization = str(
            requested_gradient_materialization
            if requested_gradient_materialization is not None
            else default_gradient_materialization
        ).strip().lower().replace("-", "_")
        if gradient_materialization not in {"full", "tiled", "streamed"}:
            raise ValueError(
                "training.gradients.materialization must be full, tiled, or streamed; "
                f"got {gradient_materialization!r}."
            )
        gradient_tile_size = int(_cfg_get(raw, "training.gradients.tile_size", 256) or 256)
        if gradient_tile_size <= 0:
            raise ValueError("training.gradients.tile_size must be a positive integer.")
        lifetime_policy = str(_cfg_get(raw, "training.memory_lifetime.policy", "separate") or "separate").strip().lower().replace("-", "_")
        if lifetime_policy not in {"separate", "phase_shared"}:
            raise ValueError(
                "training.memory_lifetime.policy must be separate or phase_shared; "
                f"got {lifetime_policy!r}."
            )
        if gradient_computation == "fused_update" and gradient_materialization != "streamed":
            raise ValueError(
                "training.gradients.computation=fused_update requires "
                "training.gradients.materialization=streamed because no parameter-gradient buffer is materialized."
            )
        if lifetime_policy == "phase_shared" and gradient_materialization == "full":
            raise ValueError(
                "training.memory_lifetime.policy=phase_shared requires "
                "training.gradients.materialization=tiled or streamed."
            )
        semantics.update({
            "requested_gradient_storage": (str(requested_gradient_storage) if gradient_storage_was_explicit else None),
            "resolved_gradient_storage": gradient_storage,
            "gradient_storage_semantics": f"parameter_gradient_{gradient_storage}",
            "parameter_gradient_computation": gradient_computation,
            "gradient_materialization": gradient_materialization,
            "gradient_materialization_tile_size": gradient_tile_size,
            "training_memory_lifetime_policy": lifetime_policy,
        })
        compile_plan.notes.update(semantics)
        memory_plan.notes.update(semantics)
        return semantics

    def _hls_weight_storage_impl(self, memory_plan=None) -> str:
        raw = self.cfg.raw
        requested = _cfg_get(
            raw,
            "memory.weight_storage",
            _cfg_get(
                raw,
                "memory.storage.weights",
                _cfg_get(raw, "training.storage.weights", "bram"),
            ),
        )
        requested = str(requested or "bram").strip().lower()

        aliases = {
            "embedded": "bram",
            "on_chip": "bram",
            "onchip": "bram",
            "block": "bram",
            "block_ram": "bram",
            "bram": "bram",
            "uram": "uram",
            "ultra": "uram",
            "ultra_ram": "uram",
            "lutram": "lutram",
            "lut_ram": "lutram",
            "distributed": "lutram",
            "ddr": "ddr",
            "external": "ddr",
            "external_ddr": "ddr",
            "dma_ddr": "ddr",
            "stream": "stream",
            "streaming": "stream",
        }
        return aliases.get(requested, "bram")
