# validate_config.py
import yaml, sys
req = {
  "libraray_arg": ["max_poly","max_dt","max_dx","use_cross_der"],
  "Noise": ["sigma_SNR"],
  "Subsampling": ["stride_x","stride_t"],
  "Dataset_name": ["equation","filename"],
  "Other_paramters": ["Tau"],
}
cfg = yaml.safe_load(open(sys.argv[1]))
missing = []
for sec, keys in req.items():
    if sec not in cfg: missing.append(f"[{sec}] section missing"); continue
    for k in keys:
        if k not in cfg[sec]: missing.append(f"[{sec}] key '{k}' missing")
if missing: 
    print("Missing:\n- " + "\n- ".join(missing)); sys.exit(1)
print("Config looks good.")
