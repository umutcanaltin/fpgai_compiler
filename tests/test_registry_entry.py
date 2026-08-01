from fpgai.registries import RegistryEntry, RegistrySource


def test_registry_entry_is_research_scoped_and_normalizes_source():
    entry = RegistryEntry(
        package_id="community.example", version="1.0.0", asset_type="model", provider="community",
        source="project-local", source_path=None, manifest_hash="sha256:abc",
        capabilities={"inference": True}, compatibility={}, validation_level="reference_tested",
        license_category="open_source", usage={"platform_scope": "research", "production_path": "morfics"}, metadata={},
    )
    assert entry.source is RegistrySource.PROJECT_LOCAL
    assert entry.usage["production_path"] == "morfics"
