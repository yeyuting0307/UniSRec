import os
import sys
import logging
from kfp import dsl
from kfp import compiler, Client
import google.cloud.aiplatform as aip

project_id = os.environ.get("PROJECT_ID")
location = os.environ.get("LOCATION")
bucket_name = os.environ.get("BUCKET_NAME")
pipeline_root = f'gs://{bucket_name}/pipeline'
package =["recbole", "accelerate", "transformers", "peft"]
image = "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime"

@dsl.component(base_image=image, packages_to_install=package)
def finetune_component(*args, **kwargs) -> str:
   import sys
   sys.path.append(f"/gcs/{bucket_name}/pipeline")
   from inprocess.warmstart.finetune import finetune
   finetune_model_path = finetune(*args, **kwargs)
   return finetune_model_path

@dsl.pipeline(
   name='finetune',
   description='Pipeline for recommendation system finetune.',
   pipeline_root=pipeline_root,
   display_name='RecSys Finetune'
)
def recsys_finetune_pipeline(*args, **kwargs) -> str:
   gpu_finetune_job_task = finetune_component(*args, **kwargs)\
      .set_cpu_limit('8') \
      .set_memory_limit('32G') \
      .add_node_selector_constraint("NVIDIA_TESLA_P100") \
      .set_accelerator_limit(1)


   return gpu_finetune_job_task.output

if __name__ == "__main__":
   compiler.Compiler().compile(recsys_finetune_pipeline, 'pipeline_finetune.yaml')

   aip.init(
      project=project_id,
      location=location,
      staging_bucket=pipeline_root
   )

   # Prepare the pipeline job
   job = aip.PipelineJob(
      display_name="recsys-pipeline-finetune",
      template_path="pipeline_finetune.yaml",
      pipeline_root=pipeline_root,
      parameter_values={},
      enable_caching = False
   )

   job.submit()
   print("Submitted")
