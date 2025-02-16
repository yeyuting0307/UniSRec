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
example_package = ["numpy", "pandas"]

# ================= Components =================
@dsl.component(base_image="python:3.9", packages_to_install=example_package)
def example_component(*args, **kwargs):
   ''' Example component '''
   import sys
   sys.path.append(f"/gcs/{bucket_name}/pipeline")
   ...


# ================= Pipeline Assemble =================
@dsl.pipeline(
   name='preprocess',
   description='Example Pipeline',
   pipeline_root=pipeline_root,
   display_name='RecSys Preprocess'
)
def example_pipeline(*args, **kwargs) -> str:
   example_task = example_component(*args, **kwargs)
   return example_task.output

if __name__ == "__main__":
   compiler.Compiler().compile(example_pipeline, 'pipeline_preprocess.yaml')

   aip.init(
      project=project_id,
      location=location,
      staging_bucket=pipeline_root
   )

   # Prepare the pipeline job
   job = aip.PipelineJob(
      display_name="recsys-preprocess-pipeline",
      template_path="pipeline_preprocess.yaml",
      pipeline_root=pipeline_root,
      parameter_values={},
      enable_caching = False
   )

   job.submit()
   print("Submitted")
