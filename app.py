from contextlib import asynccontextmanager
from typing_extensions import TypedDict

from ray import serve
from transformers import Pipeline, pipeline
from fastapi import FastAPI, Request


class AppState(TypedDict):
    pipeline: Pipeline


@asynccontextmanager
async def lifespan(app):
    pipe = pipeline("text-generation", model="facebook/opt-125m")
    app.state = AppState(pipeline=pipe)
    yield


app = FastAPI(lifespan=lifespan)


@serve.deployment(name="opt-125m")
@serve.ingress(app)
class Deployment:
    @app.get("/")
    def hello(self):
        return {"status": "running"}

    @app.get("/complete")
    def complete(self, prompt: str, request: Request):
        pipeline = request.app.state["pipeline"]

        result = pipeline(prompt)

        completion = result[0]["generated_text"]

        return {"completion": completion}


opt125 = Deployment.bind()
