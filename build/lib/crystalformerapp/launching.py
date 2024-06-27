import json
import shutil
from pathlib import Path
import os
import dp.launching.typing.addon.ui as ui
import traceback
from pprint import pprint

from dp.launching.cli import (SubParser, default_exception_handler,
                              run_sp_and_exit, to_runner)
from dp.launching.report import (AutoReportElement, MetricsChartReportElement,
                                 Report, ReportSection)
from dp.launching.typing import (BaseModel, BohriumProjectId, 
                                 BohriumUsername, Boolean, Enum, Field, Float,
                                 InputFilePath, Int, List, Literal,
                                 Optional, OutputDirectory, String, DataSet, Union)
from dp.launching.typing.addon.sysmbol import Equal


class CrystalformerOptions(BaseModel):
    spacegroup: Int = Field(ge=1, le=230, description="Space group number")
    elements: String = Field(description="Elements to include, separated by spaces")
    temperature: Float = Field(ge=0.5, le=1.5, default=1.0, description="Temperature for generation")
    seed: Int = Field(default=42, description="Random seed")


def crystalformer_runner(opts: CrystalformerOptions) -> int:
    try:
        run_crystalformer(
            spacegroup=opts.spacegroup,
            elements=opts.elements,
            temperature=opts.temperature,
            seed=opts.seed
        )
        return 0
    except Exception as exc:
        print(str(exc))
        traceback.print_exc()
        return 1

def to_parser():
    return {
        "1": SubParser(CrystalformerOptions, crystalformer_runner, "Run Crystalformer")
    }


if __name__ == '__main__':
    run_sp_and_exit(to_parser(), description="Crystal Former", version="0.1.0", exception_handler=default_exception_handler)