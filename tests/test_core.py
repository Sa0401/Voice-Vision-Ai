from visionvoice.core import AlertPlanningAgent, Detection, MobilityReasoningPipeline, RiskAssessmentAgent


def test_close_vehicle_is_prioritized_as_critical():
    agent = RiskAssessmentAgent()
    event = agent.analyze(
        [Detection("car", 0.92, (80, 210, 560, 470), track_id=4)],
        (640, 480),
    )[0]

    assert event.direction == "ahead"
    assert event.proximity == "immediate"
    assert event.severity == "critical"


def test_vehicle_requires_history_before_approaching_alert():
    agent = RiskAssessmentAgent()
    small = Detection("car", 0.90, (250, 170, 330, 260), track_id=11)
    medium = Detection("car", 0.90, (220, 140, 360, 300), track_id=11)
    large = Detection("car", 0.90, (170, 95, 420, 390), track_id=11)

    assert not agent.analyze([small], (640, 480))[0].approaching
    assert not agent.analyze([medium], (640, 480))[0].approaching
    assert agent.analyze([large], (640, 480))[0].approaching


def test_alert_planner_debounces_duplicates_but_allows_critical_realert():
    pipeline = MobilityReasoningPipeline(cooldown_seconds=5)
    assessment = pipeline.assess([Detection("truck", 0.95, (80, 120, 580, 470))], (640, 480))
    event = assessment.risks[0]
    planner = AlertPlanningAgent(cooldown_seconds=5)

    assert planner.plan([event], now=10.0)
    assert planner.plan([event], now=10.5) is None
    assert planner.plan([event], now=15.1)


def test_all_object_narration_prioritizes_hazards_and_reports_normal_scene():
    pipeline = MobilityReasoningPipeline(narrate_all=True, max_voice_objects=3)
    hazard_assessment = pipeline.assess(
        [
            Detection("person", 0.9, (200, 90, 300, 450)),
            Detection("bus", 0.9, (10, 100, 310, 440)),
            Detection("chair", 0.8, (420, 220, 570, 450)),
        ],
        (640, 480),
    )
    assert "bus" in hazard_assessment.primary_alert.lower()
    assert "safety" in hazard_assessment.primary_alert.lower()

    normal_pipeline = MobilityReasoningPipeline(narrate_all=True, max_voice_objects=2)
    normal_assessment = normal_pipeline.assess(
        [Detection("person", 0.9, (220, 90, 320, 450))],
        (640, 480),
    )
    assert normal_assessment.primary_alert.startswith("Normal scene.")


def test_all_object_mode_does_not_cap_the_spoken_list():
    pipeline = MobilityReasoningPipeline(narrate_all=True, max_voice_objects=0)
    assessment = pipeline.assess(
        [
            Detection("bus", 0.9, (10, 100, 310, 440)),
            Detection("person", 0.9, (320, 80, 390, 430)),
            Detection("chair", 0.8, (400, 220, 490, 450)),
            Detection("bottle", 0.7, (510, 270, 530, 450)),
        ],
        (640, 480),
    )
    # When max_voice_objects=0, all detected objects are included
    for obj in ["bus", "person", "chair", "bottle"]:
        assert obj in assessment.primary_alert.lower()


def test_clear_scene_is_announced_with_a_cautious_continue_message():
    pipeline = MobilityReasoningPipeline(narrate_all=True)
    assessment = pipeline.assess([], (640, 480))
    assert "normal scene" in assessment.primary_alert.lower()

