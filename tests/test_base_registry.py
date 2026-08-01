from fpgai.registries import BaseRegistry, RegistryEntry, RegistrySource


def _entry(version="1.0.0", source=RegistrySource.BUILTIN, digest="sha256:a"):
    return RegistryEntry("community.example", version, "model", "community", source, None, digest, {}, {}, "unvalidated", "research_only", {"platform_scope":"research","production_path":"morfics"}, {})


def test_registry_resolves_highest_version_then_source_priority():
    registry=BaseRegistry("model")
    registry.register(_entry("1.0.0", RegistrySource.BUILTIN, "sha256:a"))
    registry.register(_entry("1.1.0", RegistrySource.BUILTIN, "sha256:b"))
    registry.register(_entry("1.1.0", RegistrySource.PROJECT_LOCAL, "sha256:b"))
    result=registry.resolve("community.example", ">=1,<2")
    assert result.ok
    assert result.selected.version == "1.1.0"
    assert result.selected.source is RegistrySource.PROJECT_LOCAL


def test_registry_rejects_conflicting_duplicate_identity():
    registry=BaseRegistry("model")
    assert registry.register(_entry()).ok
    result=registry.register(_entry(digest="sha256:different"))
    assert not result.ok
    assert result.errors[0].code == "PKGREG003"
