from types import SimpleNamespace

from fints.client import NeedTANResponse

from cleanab import fints as fints_module
from cleanab.models.enums import AccountType


class FakeClient:
    """Stands in for FinTS3PinTanClient."""

    def __init__(self, *, mechanism="920", tan_required=False, medium=None):
        self._mechanism = mechanism
        self._tan_required = tan_required
        self.selected_tan_medium = medium
        self.mechanisms = {"920": SimpleNamespace(security_function="920", name="App")}
        self.fetched = False
        self.set_medium = None

    def get_current_tan_mechanism(self):
        return self._mechanism

    def fetch_tan_mechanisms(self):
        self.fetched = True

    def get_tan_mechanisms(self):
        return self.mechanisms

    def set_tan_mechanism(self, mech):
        self._mechanism = mech

    def is_tan_media_required(self):
        return self._tan_required

    def get_tan_media(self):
        return (None, [SimpleNamespace(tan_medium_name="Phone", mobile_number_masked="xx", last_use=None)])

    def set_tan_medium(self, medium):
        self.set_medium = medium
        self.selected_tan_medium = medium


def test_bootstrap_noop_when_mechanism_known():
    client = FakeClient()
    fints_module.bootstrap_fints(client)
    assert not client.fetched


def test_bootstrap_picks_single_tan_medium():
    client = FakeClient(tan_required=True)
    fints_module.bootstrap_fints(client)
    assert client.set_medium.tan_medium_name == "Phone"


def test_bootstrap_asks_when_multiple_mechanisms(monkeypatch):
    client = FakeClient(mechanism=None)
    client.mechanisms["921"] = SimpleNamespace(security_function="921", name="Other")
    monkeypatch.setattr("builtins.input", lambda _: "1")
    fints_module.bootstrap_fints(client)
    assert client.fetched
    assert client._mechanism == "921"


def tan_response():
    # Bypass __init__, which needs a real bank segment to parse the challenge from
    response = NeedTANResponse.__new__(NeedTANResponse)
    response.challenge = "Approve please"
    response.challenge_html = "<p>Approve please</p>"
    response.challenge_hhduc = None
    response.challenge_matrix = None
    return response


def test_handle_tan_response_success(monkeypatch):
    sent = []
    client = SimpleNamespace(send_tan=lambda response, tan: sent.append(tan) or ["tx"])
    monkeypatch.setattr("builtins.input", lambda _: "123456")
    assert fints_module.handle_tan_response(client, tan_response()) == ["tx"]
    assert sent == ["123456"]


def test_handle_tan_response_retries_on_rejected_tan(monkeypatch):
    answers = iter([tan_response(), ["tx"]])
    client = SimpleNamespace(send_tan=lambda response, tan: next(answers))
    monkeypatch.setattr("builtins.input", lambda _: "1")
    assert fints_module.handle_tan_response(client, tan_response()) == ["tx"]


def test_handle_tan_response_client_error(monkeypatch):
    def fail(response, tan):
        raise fints_module.FinTSClientError("bad")

    monkeypatch.setattr("builtins.input", lambda _: "1")
    assert fints_module.handle_tan_response(SimpleNamespace(send_tan=fail), tan_response()) == []


def test_process_fints_account_unknown_iban(account, monkeypatch):
    monkeypatch.setattr(fints_module, "get_fints_client", lambda *a: (None, [SimpleNamespace(iban="DE00")]))
    assert fints_module.process_fints_account(account, "a", "b", None) == []


def test_process_fints_account_fetches_transactions(account, monkeypatch):
    sepa = SimpleNamespace(iban=account.iban)
    monkeypatch.setattr(fints_module, "get_fints_client", lambda *a: ("client", [sepa]))
    seen = {}

    def fake_retrieve(sepa_account, client, *, start_date, end_date):
        seen.update(sepa=sepa_account, client=client, start=start_date, end=end_date)
        return ["tx"]

    monkeypatch.setattr(fints_module, "retrieve_transactions", fake_retrieve)
    assert fints_module.process_fints_account(account, "start", "end", "pid") == ["tx"]
    assert seen == {"sepa": sepa, "client": "client", "start": "start", "end": "end"}


def test_process_fints_account_holdings(account, monkeypatch):
    account.account_type = AccountType.HOLDING
    sepa = SimpleNamespace(iban=account.iban)
    monkeypatch.setattr(fints_module, "get_fints_client", lambda *a: ("client", [sepa]))
    monkeypatch.setattr(fints_module, "retrieve_holdings", lambda s, c: ["h"])
    assert fints_module.process_fints_account(account, "a", "b", None) == ["h"]
