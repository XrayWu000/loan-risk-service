import re

def validate_precision(val):
    if not val:
        return ""
    val = re.sub(r"[^\d.]", "", str(val))

    if val.count('.') > 1:
        parts = val.split('.')
        val = f"{parts[0]}.{''.join(parts[1:])}"

    if "." in val:
        parts = val.split('.')
        val = f"{parts[0]}.{parts[1][:2]}"

    try:
        num_val = float(val)
        if num_val > 99.99:
            return "99.99"
        return val
    except:
        return ""