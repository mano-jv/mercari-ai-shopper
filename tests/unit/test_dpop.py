import jwt
from mercari_shopper.sources.dpop import DpopSigner
from mercari_shopper.sources.payload import SEARCH_URL


def test_dpop_proof_header_payload_and_signature() -> None:
    signer = DpopSigner()
    token = signer.proof(SEARCH_URL, "post", now=1_700_000_000)
    header = jwt.get_unverified_header(token)
    payload = jwt.decode(token, options={"verify_signature": False})

    assert header["typ"] == "dpop+jwt"
    assert header["alg"] == "ES256"
    assert header["jwk"]["kty"] == "EC"
    assert header["jwk"]["crv"] == "P-256"
    assert header["jwk"]["x"]
    assert header["jwk"]["y"]

    assert payload["htu"] == SEARCH_URL
    assert payload["htm"] == "POST"
    assert payload["iat"] == 1_700_000_000
    assert payload["uuid"] == signer.device_uuid
    assert payload["jti"]

    key = jwt.PyJWK.from_dict(header["jwk"]).key
    verified = jwt.decode(token, key=key, algorithms=["ES256"])
    assert verified["jti"] == payload["jti"]


def test_dpop_reuses_key_and_device_uuid() -> None:
    signer = DpopSigner()
    first_token = signer.proof(SEARCH_URL, "POST")
    second_token = signer.proof(SEARCH_URL, "GET")
    first_header = jwt.get_unverified_header(first_token)
    second_header = jwt.get_unverified_header(second_token)
    first = jwt.decode(first_token, options={"verify_signature": False})
    second = jwt.decode(second_token, options={"verify_signature": False})

    assert first_header["jwk"] == second_header["jwk"]
    assert second["uuid"] == first["uuid"] == signer.device_uuid
    assert second["htm"] == "GET"
    assert second["jti"] != first["jti"]
