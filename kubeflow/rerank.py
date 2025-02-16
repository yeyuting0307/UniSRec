#%%
import os
import logging
from kfp import dsl
from kfp import compiler, Client
import google.cloud.aiplatform as aip

project_id = os.environ.get("PROJECT_ID")
location = os.environ.get("LOCATION")
bucket_name = os.environ.get("BUCKET_NAME")
pipeline_root = f'gs://{bucket_name}/pipeline'
package =["recbole", "accelerate", "transformers", "peft"]

@dsl.component(base_image="python:3.9", packages_to_install=package)
def rerank_component(*args, **kwargs) -> str:
    import sys
    sys.path.append(f"/gcs/{bucket_name}/pipeline")
    from postprocess.rerank import rerank
    result = rerank(*args, **kwargs)
    return result

@dsl.component(base_image="python:3.9", packages_to_install=package)
def cache_component(*args, **kwargs) -> str:
    import sys
    sys.path.append(f"/gcs/{bucket_name}/pipeline")
    from postprocess.rerank import caching
    result = caching(*args, **kwargs)
    return result

@dsl.pipeline(
   name='recommendation-pipeline-rerank',
   description='Pipeline for recommendation system rerank.',
   pipeline_root=pipeline_root,
   display_name='RecSys Rerank')
def recsys_rerank_pipeline(*args, **kwargs) -> str:
    rerank_job_task = rerank_component(*args, **kwargs)
    cache_job_task = cache_component(*args, **kwargs)

    cache_job_task.after(rerank_job_task)

    return rerank_job_task.output

if __name__ == "__main__":
    compiler.Compiler().compile(recsys_rerank_pipeline, 'pipeline_rerank.yaml')

    aip.init(
        project=project_id,
        location=location,
        staging_bucket=pipeline_root
    )

    job = aip.PipelineJob(
        display_name="recsys-pipeline-rerank",
        template_path="pipeline_rerank.yaml",
        pipeline_root=pipeline_root,
        parameter_values={},
        enable_caching = False
    )

    job.submit()
    print("Submitted")

# %%
