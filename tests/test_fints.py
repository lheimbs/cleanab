from types import SimpleNamespace

import pytest
from fints.client import NeedTANResponse
from fints.exceptions import FinTSConnectionError

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


def test_bootstrap_asks_again_for_invalid_choice(monkeypatch):
    client = FakeClient(mechanism=None)
    client.mechanisms["921"] = SimpleNamespace(security_function="921", name="Other")
    answers = iter(["a", "5", "1"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    fints_module.bootstrap_fints(client)
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
    with pytest.raises(fints_module.TanFailed, match="bad"):
        fints_module.handle_tan_response(SimpleNamespace(send_tan=fail), tan_response())


def test_handle_tan_response_wraps_error_once(monkeypatch):
    answers = iter([tan_response(), tan_response()])

    def send_tan(response, tan):
        answer = next(answers, None)
        if answer is None:
            raise fints_module.FinTSClientError("bad")
        return answer

    monkeypatch.setattr("builtins.input", lambda _: "1")
    with pytest.raises(fints_module.TanFailed) as raised:
        fints_module.handle_tan_response(SimpleNamespace(send_tan=send_tan), tan_response())
    assert str(raised.value) == "Failed to send TAN: bad"


def test_process_fints_account_unknown_iban(account, monkeypatch):
    monkeypatch.setattr(fints_module, "get_fints_client", lambda *a, **kw: (None, [SimpleNamespace(iban="DE00")]))
    assert fints_module.process_fints_account(account, "a", "b", None, tan_display="blocks") == []


def test_process_fints_account_fetches_transactions(account, monkeypatch):
    sepa = SimpleNamespace(iban=account.iban)
    seen = {}

    def fake_client(*args, tan_display):
        seen.update(client_args=args, tan_display=tan_display)
        return "client", [sepa]

    def fake_retrieve(sepa_account, client, *, start_date, end_date):
        seen.update(sepa=sepa_account, client=client, start=start_date, end=end_date)
        return ["tx"]

    monkeypatch.setattr(fints_module, "get_fints_client", fake_client)
    monkeypatch.setattr(fints_module, "retrieve_transactions", fake_retrieve)
    assert fints_module.process_fints_account(account, "start", "end", "pid", tan_display="blocks") == ["tx"]
    assert seen["tan_display"] == "blocks"
    assert seen["client_args"][-1] == "pid"
    assert (seen["sepa"], seen["client"], seen["start"], seen["end"]) == (sepa, "client", "start", "end")


def test_process_fints_account_holdings(account, monkeypatch):
    account.account_type = AccountType.HOLDING
    sepa = SimpleNamespace(iban=account.iban)
    monkeypatch.setattr(fints_module, "get_fints_client", lambda *a, **kw: ("client", [sepa]))
    monkeypatch.setattr(fints_module, "retrieve_holdings", lambda s, c: ["h"])
    assert fints_module.process_fints_account(account, "a", "b", None, tan_display="blocks") == ["h"]


def test_get_fints_client_cache_ignores_tan_display(monkeypatch):
    connects = []

    def fake_connect(*key, tan_display):
        connects.append((key, tan_display))
        return "client", ["sepa"]

    monkeypatch.setattr(fints_module, "_clients", {})
    monkeypatch.setattr(fints_module, "_connect", fake_connect)
    first = fints_module.get_fints_client("blz", "user", "pw", "url", "pid", tan_display="kitty")
    second = fints_module.get_fints_client("blz", "user", "pw", "url", "pid", tan_display="blocks")
    assert first is second
    assert connects == [(("blz", "user", "pw", "url", "pid"), "kitty")]


def test_get_fints_client_does_not_retry_failed_login(monkeypatch):
    connects = []

    def fake_connect(*key, tan_display):
        connects.append(key)
        raise fints_module.TanFailed("rejected")

    monkeypatch.setattr(fints_module, "_clients", {})
    monkeypatch.setattr(fints_module, "_connect", fake_connect)
    with pytest.raises(fints_module.TanFailed):
        fints_module.get_fints_client("blz", "user", "pw", "url", "pid", tan_display="blocks")
    with pytest.raises(fints_module.LoginSkipped, match="already failed.*rejected"):
        fints_module.get_fints_client("blz", "user", "pw", "url", "pid", tan_display="blocks")
    assert len(connects) == 1


def test_get_fints_client_retries_non_auth_errors(monkeypatch):
    connects = []

    def fake_connect(*key, tan_display):
        connects.append(key)
        if len(connects) == 1:
            raise FinTSConnectionError("timeout")
        return "client", []

    monkeypatch.setattr(fints_module, "_clients", {})
    monkeypatch.setattr(fints_module, "_connect", fake_connect)
    with pytest.raises(FinTSConnectionError):
        fints_module.get_fints_client("blz", "user", "pw", "url", "pid", tan_display="blocks")
    assert fints_module.get_fints_client("blz", "user", "pw", "url", "pid", tan_display="blocks") == ("client", [])


def test_get_fints_client_does_not_keep_the_exception(monkeypatch):
    def fake_connect(*key, tan_display):
        raise fints_module.TanFailed("rejected")

    clients = {}
    monkeypatch.setattr(fints_module, "_clients", clients)
    monkeypatch.setattr(fints_module, "_connect", fake_connect)
    with pytest.raises(fints_module.TanFailed):
        fints_module.get_fints_client("blz", "user", "pw", "url", "pid", tan_display="blocks")
    assert list(clients.values()) == ["rejected"]


def test_rejected_tan_after_login_skips_other_accounts(account, monkeypatch):
    sepa = SimpleNamespace(iban=account.iban)
    connects = []

    def fake_connect(*key, tan_display):
        connects.append(key)
        return "client", [sepa]

    def rejected(*args, **kwargs):
        raise fints_module.TanFailed("rejected")

    monkeypatch.setattr(fints_module, "_clients", {})
    monkeypatch.setattr(fints_module, "_connect", fake_connect)
    monkeypatch.setattr(fints_module, "retrieve_transactions", rejected)
    with pytest.raises(fints_module.TanFailed):
        fints_module.process_fints_account(account, "a", "b", "pid", tan_display="blocks")
    with pytest.raises(fints_module.LoginSkipped, match="rejected"):
        fints_module.process_fints_account(account, "a", "b", "pid", tan_display="blocks")
    assert len(connects) == 1


def test_get_fints_client_login_without_accounts_is_not_a_failure(monkeypatch):
    connects = []

    def fake_connect(*key, tan_display):
        connects.append(key)
        return "client", []

    monkeypatch.setattr(fints_module, "_clients", {})
    monkeypatch.setattr(fints_module, "_connect", fake_connect)
    assert fints_module.get_fints_client("blz", "user", "pw", "url", "pid", tan_display="blocks") == ("client", [])
    assert fints_module.get_fints_client("blz", "user", "pw", "url", "pid", tan_display="blocks") == ("client", [])
    assert len(connects) == 1


def test_client_carries_tan_display():
    client = fints_module.CleanabFinTSClient(
        bank_identifier="12345678",
        user_id="u",
        pin="p",
        server="https://bank.invalid",
        product_id="x",
        tan_display="iterm",
    )
    assert client.tan_display == "iterm"
