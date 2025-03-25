from io import BytesIO
from fastapi import FastAPI
from fastapi.responses import Response
import torch

from ray import serve
from ray.serve.handle import DeploymentHandle


app = FastAPI()


@serve.deployment(name="APIIngress")
@serve.ingress(app)
class APIIngress:
    def __init__(self, diffusion_model_handle: DeploymentHandle) -> None:
        self.handle = diffusion_model_handle

    @app.get(
        "/imagine",
        responses={200: {"content": {"image/png": {}}}},
        response_class=Response,
    )
    async def generate(self, prompt: str, img_size: int = 512):
        assert len(prompt), "prompt parameter cannot be empty"

        image = await self.handle.generate.remote(prompt, img_size=img_size)
        file_stream = BytesIO()
        image.save(file_stream, "PNG")
        return Response(content=file_stream.getvalue(), media_type="image/png")


@serve.deployment(name="AnythingXL")
class AnythingXL:
    def __init__(self):
        from diffusers import StableDiffusionXLPipeline

        model_id = "eienmojiki/Anything-XL"

        self.pipe = StableDiffusionXLPipeline.from_pretrained(model_id)
        self.pipe = self.pipe.to("cuda")

    async def generate(self, prompt: str, img_size: int = 512):
        assert len(prompt), "prompt parameter cannot be empty"

        with torch.autocast("cuda"):
            image = self.pipe(prompt, height=img_size, width=img_size).images[0]
            return image


anything_v5 = APIIngress.bind(AnythingXL.bind())
