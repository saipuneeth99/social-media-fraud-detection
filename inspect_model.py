import joblib
import os

outdir = "outputs"
best_model_path = os.path.join(outdir, "best_model_GradientBoosting.joblib")
pipeline_path = os.path.join(outdir, "pipeline_GradientBoosting.joblib")

print(f"--- Inspecting {best_model_path} ---")
try:
    best_model = joblib.load(best_model_path)
    print(f"Type: {type(best_model)}")
    if hasattr(best_model, "steps"):
        print("Steps:", [s[0] for s in best_model.steps])
    else:
        print("No steps attribute (likely not a Pipeline)")
except Exception as e:
    print(f"Error loading best_model: {e}")

print(f"\n--- Inspecting {pipeline_path} ---")
try:
    pipeline = joblib.load(pipeline_path)
    print(f"Type: {type(pipeline)}")
    if hasattr(pipeline, "steps"):
        print("Steps:", [s[0] for s in pipeline.steps])
    else:
        print("No steps attribute (likely not a Pipeline)")
except Exception as e:
    print(f"Error loading pipeline: {e}")
