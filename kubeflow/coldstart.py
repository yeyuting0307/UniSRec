import os
from kfp import dsl
from kfp import compiler, Client
import google.cloud.aiplatform as aip

project_id = os.environ.get("PROJECT_ID")
location = os.environ.get("LOCATION")
bucket_name = os.environ.get("BUCKET_NAME")
pipeline_root = f'gs://{bucket_name}/pipeline'
package =["recbole", "accelerate", "transformers", "peft"]
image = "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime"

# ================= Components =================
@dsl.component(base_image=image, packages_to_install=package)
def coldstart_inference_component(*args, **kwargs) -> str:
   import sys
   sys.path.append(f"/gcs/{bucket_name}/pipeline")
   from inprocess.coldstart.inference import coldstart_inference
   coldstart_inference(*args, **kwargs)
   return "Done"


# ================= Pipeline Assemble =================
@dsl.pipeline(
   name='coldstart',
   description='Pipeline for recommendation system coldstart.',
   pipeline_root=pipeline_root,
   display_name='RecSys ColdStart'
)
def recsys_coldstart_pipeline(*args, **kwargs) -> str:
   coldstart_inference_task = coldstart_inference_component(*args, **kwargs)\
   .set_cpu_limit('16')\
   .set_memory_limit('64G')\
   .add_node_selector_constraint("NVIDIA_TESLA_P100") \
   .set_accelerator_limit(2)

   return coldstart_inference_task.output

#%%
if __name__ == "__main__":
   compiler.Compiler().compile(recsys_coldstart_pipeline, 'pipeline_coldstart.yaml')

   aip.init(
      project=project_id,
      location=location,
      staging_bucket=pipeline_root
   )

   job = aip.PipelineJob(
      display_name="recsys-coldstart-pipeline",
      template_path="pipeline_coldstart.yaml",
      pipeline_root=pipeline_root,
      parameter_values={},
      enable_caching = False
   )

   job.submit()
   print("Submitted")
