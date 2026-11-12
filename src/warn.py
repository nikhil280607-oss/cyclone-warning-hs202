"""Warning module: turn the model's RI flag into an SMS warning for fishers.

    storm record at time T -> RI model -> alert rule -> coast at risk -> language -> SMS

Where the storm is heading comes from IMD's official track forecast. In the replay
we use the storm's real track as a stand-in for that forecast.
"""
import json, math
import numpy as np
from matplotlib.path import Path

NEAR_KM = 300        # a coast is "at risk" if the track comes within 300 km of it
WINDOW_H = 72        # send only in the last 72 h before arrival (IMD's first warning stage starts at 72 h)

# ---------- 1. Which language for which coast ----------
LANGUAGE = {"Odisha": "Odia", "West Bengal": "Bengali", "Gujarat": "Gujarati",
            "Dadra and Nagar Haveli and Daman and Diu": "Gujarati", "Tamil Nadu": "Tamil", "Puducherry": "Tamil",
            "Andhra Pradesh": "Telugu", "Maharashtra": "Marathi", "Kerala": "Malayalam", "Karnataka": "Kannada"}
# Goa: English only for now
REGION = {"Tamil Nadu": "தமிழ்நாடு", "Puducherry": "புதுச்சேரி", "Andhra Pradesh": "ఆంధ్రప్రదేశ్",
          "Maharashtra": "महाराष्ट्राच्या", "Kerala": "കേരള", "Karnataka": "ಕರ್ನಾಟಕ"}

# first sentence: "strengthening fast" when the model flags RI now, "still dangerous" for later updates
FIRST = {
    "growing": {"English": "CYCLONE {name} is strengthening fast.", "Odia": "ଘୂର୍ଣ୍ଣିବାତ୍ୟା {name} ଦ୍ରୁତ ଗତିରେ ଶକ୍ତିଶାଳୀ ହେଉଛି।",
                "Bengali": "ঘূর্ণিঝড় {name} দ্রুত শক্তিশালী হচ্ছে।", "Gujarati": "વાવાઝોડું {name} ઝડપથી મજબૂત બની રહ્યું છે.",
                "Tamil": "{name} புயல் வேகமாக வலுவடைந்து வருகிறது.", "Telugu": "{name} తుఫాను వేగంగా బలపడుతోంది.",
                "Marathi": "{name} चक्रीवादळ वेगाने तीव्र होत आहे.", "Malayalam": "{name} ചുഴലിക്കാറ്റ് അതിവേഗം ശക്തിപ്രാപിക്കുന്നു.",
                "Kannada": "{name} ಚಂಡಮಾರುತ ವೇಗವಾಗಿ ಬಲಗೊಳ್ಳುತ್ತಿದೆ."},
    "still":   {"English": "CYCLONE {name} is still dangerous.", "Odia": "ଘୂର୍ଣ୍ଣିବାତ୍ୟା {name} ଏବେ ବି ବିପଜ୍ଜନକ।",
                "Bengali": "ঘূর্ণিঝড় {name} এখনও বিপজ্জনক।", "Gujarati": "વાવાઝોડું {name} હજુ પણ ખતરનાક છે.",
                "Tamil": "{name} புயல் இன்னும் ஆபத்தானதாக உள்ளது.", "Telugu": "{name} తుఫాను ఇంకా ప్రమాదకరంగానే ఉంది.",
                "Marathi": "{name} चक्रीवादळ अजूनही धोकादायक आहे.", "Malayalam": "{name} ചുഴലിക്കാറ്റ് ഇപ്പോഴും അപകടകരമാണ്.",
                "Kannada": "{name} ಚಂಡಮಾರುತ ಇನ್ನೂ ಅಪಾಯಕಾರಿಯಾಗಿದೆ."},
}
# the rest: action first, then where and when (from IMD's track forecast)
REST = {
    "English":  "Do not go to sea. Boats at sea: return to shore now. Expected near {state} coast around {date}.",
    "Odia":     "ସମୁଦ୍ରକୁ ଯାଆନ୍ତୁ ନାହିଁ। ସମୁଦ୍ରରେ ଥିବା ଡଙ୍ଗା ତୁରନ୍ତ କୂଳକୁ ଫେରନ୍ତୁ। {date} ନାଗାଦ ଓଡ଼ିଶା ଉପକୂଳ ନିକଟରେ ପହଞ୍ଚିବା ସମ୍ଭାବନା।",
    "Bengali":  "সমুদ্রে যাবেন না। সমুদ্রে থাকলে এখনই তীরে ফিরে আসুন। {date} নাগাদ পশ্চিমবঙ্গ উপকূলের কাছে পৌঁছাতে পারে।",
    "Gujarati": "દરિયામાં ન જશો. દરિયામાં હો તો તરત કિનારે પાછા ફરો. {date} આસપાસ ગુજરાતના દરિયાકિનારા નજીક પહોંચવાની શક્યતા.",
    "Tamil":    "கடலுக்குச் செல்ல வேண்டாம். கடலில் இருப்பவர்கள் உடனே கரைக்குத் திரும்புங்கள். {date} வாக்கில் {region} கடற்கரையை நெருங்கக்கூடும்.",
    "Telugu":   "సముద్రంలోకి వెళ్లవద్దు. సముద్రంలో ఉన్నవారు వెంటనే తీరానికి తిరిగి రండి. {date} నాటికి {region} తీరానికి చేరువయ్యే అవకాశం ఉంది.",
    "Marathi":  "समुद्रात जाऊ नका. समुद्रात असाल तर लगेच किनाऱ्यावर परत या. {date}च्या सुमारास {region} किनाऱ्याजवळ पोहोचण्याची शक्यता.",
    "Malayalam": "കടലിൽ പോകരുത്. കടലിലുള്ളവർ ഉടൻ കരയിലേക്ക് മടങ്ങുക. {date}-ഓടെ {region} തീരത്തിന് സമീപം എത്താൻ സാധ്യത.",
    "Kannada":  "ಸಮುದ್ರಕ್ಕೆ ಹೋಗಬೇಡಿ. ಸಮುದ್ರದಲ್ಲಿದ್ದರೆ ತಕ್ಷಣ ದಡಕ್ಕೆ ಮರಳಿ. {date} ರ ವೇಳೆಗೆ {region} ಕರಾವಳಿಯ ಸಮೀಪ ತಲುಪುವ ಸಾಧ್ಯತೆ ಇದೆ.",
}
DIGITS = {"Odia": "୦୧୨୩୪୫୬୭୮୯", "Bengali": "০১২৩৪৫৬৭৮৯", "Gujarati": "૦૧૨૩૪૫૬૭૮૯", "Marathi": "०१२३४५६७८९"}
MONTH_FIRST = {"Tamil", "Telugu", "Malayalam", "Kannada"}      # "November 26" order
MONTHS = {
    "English": "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(),
    "Odia": "ଜାନୁଆରୀ ଫେବୃଆରୀ ମାର୍ଚ୍ଚ ଅପ୍ରେଲ ମଇ ଜୁନ ଜୁଲାଇ ଅଗଷ୍ଟ ସେପ୍ଟେମ୍ବର ଅକ୍ଟୋବର ନଭେମ୍ବର ଡିସେମ୍ବର".split(),
    "Bengali": "জানুয়ারি ফেব্রুয়ারি মার্চ এপ্রিল মে জুন জুলাই আগস্ট সেপ্টেম্বর অক্টোবর নভেম্বর ডিসেম্বর".split(),
    "Gujarati": "જાન્યુઆરી ફેબ્રુઆરી માર્ચ એપ્રિલ મે જૂન જુલાઈ ઑગસ્ટ સપ્ટેમ્બર ઑક્ટોબર નવેમ્બર ડિસેમ્બર".split(),
    "Tamil": "ஜனவரி பிப்ரவரி மார்ச் ஏப்ரல் மே ஜூன் ஜூலை ஆகஸ்ட் செப்டம்பர் அக்டோபர் நவம்பர் டிசம்பர்".split(),
    "Telugu": "జనవరి ఫిబ్రవరి మార్చి ఏప్రిల్ మే జూన్ జూలై ఆగస్టు సెప్టెంబర్ అక్టోబర్ నవంబర్ డిసెంబర్".split(),
    "Marathi": "जानेवारी फेब्रुवारी मार्च एप्रिल मे जून जुलै ऑगस्ट सप्टेंबर ऑक्टोबर नोव्हेंबर डिसेंबर".split(),
    "Malayalam": "ജനുവരി ഫെബ്രുവരി മാർച്ച് ഏപ്രിൽ മേയ് ജൂൺ ജൂലൈ ഓഗസ്റ്റ് സെപ്റ്റംബർ ഒക്ടോബർ നവംബർ ഡിസംബർ".split(),
    "Kannada": "ಜನವರಿ ಫೆಬ್ರವರಿ ಮಾರ್ಚ್ ಏಪ್ರಿಲ್ ಮೇ ಜೂನ್ ಜುಲೈ ಆಗಸ್ಟ್ ಸೆಪ್ಟೆಂಬರ್ ಅಕ್ಟೋಬರ್ ನವೆಂಬರ್ ಡಿಸೆಂಬರ್".split(),
}
# storm names in local script (storms that came near these coasts, 2013-2025)
LOCAL_NAME = {
    "Odia": {"FANI": "ଫନି", "BULBUL": "ବୁଲବୁଲ", "AMPHAN": "ଆମ୍ଫାନ", "YAAS": "ୟାସ", "GULAB": "ଗୁଲାବ",
             "JAWAD": "ଜୱାଦ", "ASANI": "ଅସନି", "SITRANG": "ସିତ୍ରାଙ୍ଗ", "DANA": "ଦାନା", "REMAL": "ରେମାଲ",
             "HAMOON": "ହାମୁନ", "MIDHILI": "ମିଧିଲି", "MOCHA": "ମୋଚା", "PHAILIN": "ଫାଇଲିନ", "HUDHUD": "ହୁଦହୁଦ",
             "TITLI": "ତିତଲି"},
    "Bengali": {"FANI": "ফণী", "BULBUL": "বুলবুল", "AMPHAN": "আমফান", "YAAS": "ইয়াস", "GULAB": "গুলাব",
                "JAWAD": "জাওয়াদ", "ASANI": "অশনি", "SITRANG": "সিত্রাং", "DANA": "দানা", "REMAL": "রেমাল",
                "HAMOON": "হামুন", "MIDHILI": "মিধিলি", "MOCHA": "মোখা", "ROANU": "রোয়ানু"},
    "Gujarati": {"VAYU": "વાયુ", "TAUKTAE": "તૌકતે", "BIPARJOY": "બિપરજોય", "MAHA": "મહા", "KYAAR": "ક્યાર",
                 "SHAHEEN": "શાહીન", "TEJ": "તેજ", "ASNA": "અસના", "NISARGA": "નિસર્ગ", "HIKKA": "હિક્કા"},
    "Tamil": {"NIVAR": "நிவர்", "BUREVI": "புரெவி", "MANDOUS": "மாண்டஸ்", "FENGAL": "ஃபெஞ்சல்", "DITWAH": "டிட்வா",
              "GAJA": "கஜா", "VARDAH": "வர்தா", "MICHAUNG": "மிக்ஜாம்", "OCKHI": "ஒக்கி"},
    "Telugu": {"MICHAUNG": "మిచౌంగ్", "MONTHA": "మొంథా", "ASANI": "అసని", "GULAB": "గులాబ్", "HUDHUD": "హుద్‌హుద్",
               "TITLI": "తిత్లీ", "PHETHAI": "పెథాయ్", "JAWAD": "జవాద్", "ROANU": "రోవాను"},
    "Marathi": {"NISARGA": "निसर्ग", "TAUKTAE": "तौक्ते", "KYAAR": "क्यार", "BIPARJOY": "बिपरजॉय", "VAYU": "वायू"},
    "Malayalam": {"OCKHI": "ഓഖി", "TAUKTAE": "ടൗട്ടേ"},
    "Kannada": {"TAUKTAE": "ತೌಕ್ತೆ", "OCKHI": "ಓಖಿ"},
}


def local_date(t_ist, lang):
    day = str(t_ist.day)
    if lang in DIGITS:
        day = "".join(DIGITS[lang][int(c)] for c in day)
    month = MONTHS[lang][t_ist.month - 1]
    return f"{month} {day}" if lang in MONTH_FIRST else f"{day} {month}"


def make_sms(name, state, t_arrive_utc, kind="growing"):
    """The warning text: English always, plus the coast's own language when we have it."""
    t_ist = t_arrive_utc.tz_convert("Asia/Kolkata")
    name = name.upper() if isinstance(name, str) else ""          # unnamed depressions: just "CYCLONE"
    langs = ["English"] + ([LANGUAGE[state]] if state in LANGUAGE else [])
    out = {}
    for lang in langs:
        nm = name if lang == "English" else LOCAL_NAME[lang].get(name, name.title())
        text = FIRST[kind][lang].format(name=nm) + " " + REST[lang].format(
            state=state, date=local_date(t_ist, lang), region=REGION.get(state, ""))
        out[lang] = " ".join(text.split())
    return out


# ---------- 2. How many SMS a text needs ----------
GSM7 = set("@£$¥èéùìòÇ\nØø\rÅåΔ_ΦΓΛΩΠΨΣΘΞÆæßÉ !\"#¤%&'()*+,-./0123456789:;<=>?¡"
           "ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÑÜ§¿abcdefghijklmnopqrstuvwxyzäöñüà")
GSM7_EXT = set("^{}\\[~]|€")


def sms_parts(text):
    """Plain English fits 160 per SMS (153 if split). Indian scripts need Unicode: 70 (67 if split)."""
    if all(c in GSM7 or c in GSM7_EXT for c in text):
        n = sum(2 if c in GSM7_EXT else 1 for c in text)
        return n, (1 if n <= 160 else math.ceil(n / 153)), "GSM-7"
    n = len(text.encode("utf-16-le")) // 2
    return n, (1 if n <= 70 else math.ceil(n / 67)), "Unicode"


# ---------- 3. Which Indian coast is at risk ----------
class Coast:
    def __init__(self, geojson):
        self.states = []
        for f in json.load(open(geojson))["features"]:
            g = f["geometry"]
            polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
            rings = [np.array(p[0]) for p in polys]
            self.states.append((f["properties"]["name"], [Path(r) for r in rings], np.vstack(rings)))

    def nearest(self, lat, lon):
        """(state, km): 0 km if the point is inside a coastal state."""
        best = (None, 1e9)
        for name, paths, pts in self.states:
            if any(p.contains_point((lon, lat)) for p in paths):
                return name, 0.0
            km = _km(lat, lon, pts[:, 1], pts[:, 0]).min()
            if km < best[1]:
                best = (name, km)
        return best

    def at_risk(self, track):
        """Target state and arrival time: first entry into a coastal state, else closest approach (<= NEAR_KM).
        A storm whose closest point is its very first record is moving away from India: no coast at risk."""
        near = [self.nearest(a, b) for a, b in zip(track.lat, track.lon)]
        km = np.array([n[1] for n in near])
        inside = np.where(km == 0)[0]
        i = inside[0] if len(inside) else int(km.argmin())
        if km[i] > NEAR_KM or i == 0:
            return None, None, float(km[i])
        return near[i][0], track.t.iloc[i], float(km[i])


def _km(lat1, lon1, lat2, lon2):
    r = np.radians
    a = np.sin(r(lat2 - lat1) / 2) ** 2 + np.cos(r(lat1)) * np.cos(r(lat2)) * np.sin(r(lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


# ---------- 4. The alert rule ----------
def warnings_for_storm(storm, flags, coast):
    """storm: rows of one storm in time order; flags: 0/1 RI flag from the model at each row.
    The alert switches ON at the first RI flag and stays on while the storm is cyclone strength (34 kt+).
    SMS go out only in the 72 h before the storm reaches the coast: the first one as soon as the
    alert is on, then at most one update every 24 h.
    Returns (state, arrival time, SMS sent, number of 3-hourly records while the alert was on)."""
    state, arrive, _ = coast.at_risk(storm)
    if state is None:                    # not heading for India
        return None, None, [], 0
    sent, alert_on, updates = [], False, 0
    for (_, row), flag in zip(storm.iterrows(), flags):
        # ON at the first RI flag, stays on while 34 kt+
        alert_on = bool(flag) or (alert_on and row.wind >= 34)
        left_h = (arrive - row.t).total_seconds() / 3600
        if not alert_on or not 0 < left_h <= WINDOW_H:
            continue
        updates += 1                     # an IMD update
        if not sent or hours(row.t - sent[-1]["t"]) >= 24:
            kind = "growing" if flag else "still"
            sms = make_sms(row["name"], state, arrive, kind)
            sent.append({"t": row.t, "hours_before_arrival": left_h,
                         "kind": kind, "sms": sms})
    return state, arrive, sent, updates


def hours(dt):
    return dt.total_seconds() / 3600
