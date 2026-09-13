import os
from pathlib import Path

from pydantic_ai import Agent, BinaryContent, RunContext
from pydantic import BaseModel, Field
from pydantic_ai.usage import UsageLimits




class XRDPeak(BaseModel):
    peak_position: float = Field(description="Peak position in 2θ (degrees)")
    intensity: float = Field(description="Peak intensity")
    other_details: str | None = Field(description="Likely phase or plane present", default=None)


class XRDAnalysisResult(BaseModel):
    peaks: list[XRDPeak]
    background_noise_level: str = Field(description="High, medium, or low background noise level")
    phase_identified: str = Field(description="Likely phase(s) in the sample", default=None)
    crystallinity : str = Field(description="Crystalline, semicrystalline, or amorphous with reasoning")
    summary: str = Field(description="Plain text summary of the analysis")


guilVision = Agent(
    os.getenv("GUILLEMOT_AI_MODEL"),
    name="guilVision",
    description="Analyzes powder X-ray Diffraction (XRD) images/plots and return the information in a structured text format. ",
    output_type= XRDAnalysisResult,
    system_prompt= """You are a powder X-ray diffraction 
    (PXRD) analyst. You are given an image of a PXRD plot (2-theta on x-axis, intensity on y-axis).
 
    Read the image directly. Extract:
    - approximate 2-theta positions and relative intensities of major peaks
    - background noise level
    - likely phase(s) present (if identifiable from the peaks)
    - crystallinity assessment (sharp peaks = crystalline, broad humps = amorphous)
    - a short plain-text summary 
 
    Do not invent precise values you cannot read from the plot. State
    approximate values and flag low confidence where the image is unclear.""", 
)

GUILVISION_USAGE_LIMITS= UsageLimits(
    total_tokens_limit= 10000,
    request_limit= 10,
)


@guilVision.tool
async def analyze_xrd_image(ctx= RunContext, image_path: str|None=None) -> str:
    """
    Analyze a PXRD plot image and return a text report.

    Call this whenever the user provides or references a PXRD plot image file.
    image_path: local filesystem path to the image (png/jpg).
    """
    data= Path(image_path).read_bytes()
    media_type= "image/png" if image_path.lower().endswith(".png") else "image/jpeg"


    result = await guilVision.run(
        [
            "Analyze this PXRD plot",
            BinaryContent(data=data, media_type=media_type),
        ],

        usage_limits= GUILVISION_USAGE_LIMITS,
        
    )

    analysis = result.output
    peak_lines = "\n".join(
        f"- 2θ={p.two_theta:.2f}°, rel. intensity={p.relative_intensity:.0f}%"
        + (f", assignment={p.assignment}" if p.assignment else "")
        for p in analysis.peaks
    )

    return (
        f"PXRD Analysis Report:\n"
        f"Background noise level: {analysis.background_noise_level}\n"
        f"Likely phase(s) identified: {analysis.phase_identified or'none identified'}\n"
        f"Crystallinity assessment: {analysis.crystallinity}\n"
        f"Peak details:\n{peak_lines}\n"
        f"Summary: {analysis.summary}"
    )




if __name__ == "__main__":

    result = guilVision.run_sync(
        "Analyze this PXRD plot at ./example_xrd_plot.png",
    )
    print(result.output)