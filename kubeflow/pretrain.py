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
def pretrain_component(*args, **kwargs) -> str:
   import sys
   sys.path.append(f"/gcs/{bucket_name}/pipeline")
   from inprocess.warmstart.pretrain import pretrain
   pretrain_model_path = pretrain(*args, **kwargs)
   return pretrain_model_path

@dsl.component(base_image=image, packages_to_install=package)
def finetune_component(pretrain_model_path:str, *args, **kwargs) -> str:
   import sys
   sys.path.append(f"/gcs/{bucket_name}/pipeline")
   from inprocess.warmstart.finetune import finetune

   finetune_model_path = finetune(pretrain_model_path, *args, **kwargs)
   return finetune_model_path


@dsl.pipeline(
   name='recommendation-pipeline-pretrain',
   description='Pipeline for recommendation system pretrain.',
   pipeline_root=pipeline_root,
   display_name='RecSys Pretrain'
)
def recsys_pretrain_pipeline(*args, **kwargs) -> str:
   gpu_pretrain_job_task = pretrain_component(*args, **kwargs)\
      .set_cpu_limit('32') \
      .set_memory_limit('128G') \
      .add_node_selector_constraint("NVIDIA_TESLA_P100") \
      .set_accelerator_limit(4)
   
   gpu_finetune_job_task = finetune_component(
         remote_pretrain_model_path = gpu_pretrain_job_task.output
      ).set_cpu_limit('8') \
      .set_memory_limit('32G') \
      .add_node_selector_constraint("NVIDIA_TESLA_P100") \
      .set_accelerator_limit(1)
      
   return gpu_finetune_job_task.output


if __name__ == "__main__":
   compiler.Compiler().compile(recsys_pretrain_pipeline, 'pipeline_pretrain.yaml')

   aip.init(
      project=project_id,
      location=location,
      staging_bucket=pipeline_root
   )

   # Prepare the pipeline job
   job = aip.PipelineJob(
      display_name="recsys-pipeline-pretrain",
      template_path="pipeline_pretrain.yaml",
      pipeline_root=pipeline_root,
      parameter_values={},
      enable_caching = False
   )

   job.submit()
   print("Submitted")
   