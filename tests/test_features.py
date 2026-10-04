from features import FEATURE_NAMES, extract_one


def feats(text):
    return dict(zip(FEATURE_NAMES, extract_one(text)))


def test_feature_vector_length_matches_names():
    assert len(extract_one("hello")) == len(FEATURE_NAMES)


def test_detects_link():
    assert feats("Update KYC at http://sbi-kyc-update.in")["has_link"] == 1.0
    assert feats("See you at the library tomorrow")["has_link"] == 0.0


def test_detects_indian_phone_number():
    assert feats("Call 9876543210 now")["has_phone_number"] == 1.0
    assert feats("I will come at 5")["has_phone_number"] == 0.0


def test_detects_money_and_urgency():
    f = feats("Pay Rs 49 immediately or your account will be blocked today")
    assert f["mentions_money"] == 1.0
    assert f["urgent_words"] >= 2


def test_empty_text_does_not_crash():
    assert len(extract_one("")) == len(FEATURE_NAMES)
    assert len(extract_one(None)) == len(FEATURE_NAMES)
