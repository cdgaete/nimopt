import io

from nimopt.models import storage
from nimopt.progress import Progress, reporter


class Recorded:
    """A reporter that keeps what it was told."""

    def __init__(self):
        self.calls = []

    def start(self, total, what):
        self.calls.append(("start", total, what))

    def step(self, done, what):
        self.calls.append(("step", done, what))

    def done(self):
        self.calls.append(("done",))


class Terminal(io.StringIO):
    """A stream that reports itself as a terminal."""

    def isatty(self):
        return True


def test_false_reports_nothing_and_true_reports_through_the_built_in():
    assert reporter(False) is None
    assert reporter(None) is None
    assert isinstance(reporter(True), Progress)


def test_a_reporter_of_the_callers_own_is_taken_as_it_is():
    held = Recorded()
    assert reporter(held) is held


def test_a_report_to_a_pipe_writes_nothing():
    # a redirected run writes the same text, with no carriage returns and no
    # bars in the middle of it
    stream = io.StringIO()
    bar = Progress(stream)
    bar.start(100, "building m")
    bar.step(50, "cap")
    bar.done()
    assert stream.getvalue() == ""


def test_a_report_to_a_terminal_states_the_work_and_the_fraction():
    stream = Terminal()
    bar = Progress(stream)
    bar.start(100, "building m")
    bar.step(63, "supply_balance")
    bar.done()
    written = stream.getvalue()
    assert "building m" in written
    assert "supply_balance" in written
    assert "63" in written
    assert written.endswith("\n")


def test_a_pass_that_states_no_total_states_no_fraction():
    stream = Terminal()
    bar = Progress(stream)
    bar.start(None, "building m")
    bar.step(1, "cap")
    bar.done()
    assert "%" not in stream.getvalue()


def test_assembling_reports_every_nonzero_the_model_carries():
    held = Recorded()
    model = storage.definition().build(storage.data(2))
    model.assemble(progress=held)
    assert held.calls[0] == ("start", model.nnz, f"assembling {model.name}")
    assert held.calls[-1] == ("done",)
    steps = [c for c in held.calls if c[0] == "step"]
    assert steps, "a model of seven constraints reports steps"
    assert steps[-1][1] == model.nnz, "the last step accounts for every nonzero"
    assert [c[1] for c in steps] == sorted(c[1] for c in steps), "steps ascend"


def test_a_step_names_the_constraint_it_finished():
    held = Recorded()
    model = storage.definition().build(storage.data(2))
    model.assemble(progress=held)
    named = {c[2] for c in held.calls if c[0] == "step"}
    assert named & set(model.constraints), named


def test_assembling_without_a_reporter_reports_nothing():
    model = storage.definition().build(storage.data(2))
    assembled = model.assemble()
    assert assembled.values.size == model.nnz


def test_building_reports_the_constraints_it_measures():
    held = Recorded()
    definition = storage.definition()
    model = definition.build(storage.data(2), progress=held)
    assert held.calls[0] == ("start", len(definition.constraints), "building storage")
    assert held.calls[-1] == ("done",)
    steps = [c for c in held.calls if c[0] == "step"]
    assert len(steps) == len(definition.constraints)
    assert steps[-1][1] == len(definition.constraints)
    assert set(model.constraints) == {c[2] for c in steps}


def test_building_without_a_reporter_reports_nothing():
    model = storage.definition().build(storage.data(2))
    assert model.nnz > 0


def test_solving_reports_the_assembly_it_did():
    held = Recorded()
    model = storage.definition().build(storage.data(2))
    model.solve(progress=held)
    assert held.calls[0][0] == "start"
    assert held.calls[0][2] == f"assembling {model.name}"
    assert held.calls[-1] == ("done",)


def test_the_report_finishes_before_a_solver_writes_a_line():
    # nothing interleaves: the report covers building, which is done before
    # a solver starts
    held = Recorded()
    model = storage.definition().build(storage.data(2))
    session = model.session(progress=held)
    assert held.calls[-1] == ("done",)
    session.close()
