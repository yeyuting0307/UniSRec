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
image = "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime"

@dsl.component(base_image=image, packages_to_install=package)
def inference_component(*args, **kwargs) -> str:
    import sys
    sys.path.append(f"/gcs/{bucket_name}/pipeline")
    from inprocess.warmstart.inference import predict
    result = predict(*args, **kwargs)
    return result


@dsl.pipeline(
   name='inference',
   description='Pipeline for recommendation system inference.',
   pipeline_root=pipeline_root,
   display_name='RecSys Inference')
def recsys_inference_pipeline() -> str:
    inference_job_task = inference_component(
        project_id=project_id, 
        location=location, 
        bucket_name=bucket_name
        ).set_cpu_limit('8') \
        .set_memory_limit('32G')

    return inference_job_task.output

if __name__ == "__main__":
    compiler.Compiler().compile(recsys_inference_pipeline, 'pipeline_inference.yaml')

    aip.init(
        project=project_id,
        location=location,
        staging_bucket=pipeline_root
    )

    job = aip.PipelineJob(
        display_name="recsys-pipeline-inference",
        template_path="pipeline_inference.yaml",
        pipeline_root=pipeline_root,
        parameter_values={},
        enable_caching = False
    )

    job.submit()
    print("Submitted")

# %%
