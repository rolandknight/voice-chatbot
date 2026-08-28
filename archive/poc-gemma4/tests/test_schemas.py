from poc_gemma4.schemas import load_skills, to_openai_tools


def test_loads_every_skill_sorted(skills_root):
    skills = load_skills(skills_root)
    names = [s["name"] for s in skills]
    assert names == sorted(names)
    assert {"get_current_time", "play_bbc_radio", "set_timer", "switch_persona"} <= set(names)


def test_enabled_when_filters(skills_root, cfg):
    all_ = load_skills(skills_root)
    off = load_skills(skills_root, {k: False for k in cfg["skills"]["enabled"]})
    assert len(off) < len(all_)
    assert all(s["enabled_when"] is None for s in off)


def test_openai_shape_matches_loader(skills_root):
    tools = to_openai_tools(load_skills(skills_root))
    radio = next(t for t in tools if t["function"]["name"] == "play_bbc_radio")
    assert radio["type"] == "function"
    params = radio["function"]["parameters"]
    assert params["type"] == "object"
    assert params["required"] == ["station"]
    assert params["properties"]["station"]["type"] == "string"
    assert "\n" not in radio["function"]["description"]


def test_tool_order_is_stable_regardless_of_input_order(skills_root):
    skills = load_skills(skills_root)
    a = to_openai_tools(skills)
    b = to_openai_tools(list(reversed(skills)))
    assert a == b
