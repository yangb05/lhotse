import click

from lhotse.bin.modes import prepare
from lhotse.recipes.sada import prepare_sada
from lhotse.utils import Pathlike


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("utterance_dir", type=click.Path())
@click.argument("output_dir", type=click.Path())
@click.option(
    "-j",
    "--num-jobs",
    type=int,
    default=1,
    help="How many threads to use (can give good speed-ups with slow disks).",
)
def sada(
    corpus_dir: Pathlike,
    utterance_dir: Pathlike,
    output_dir: Pathlike,
    num_jobs: int,
):
    """
    The sada corpus preparation.
    """
    prepare_sada(
        corpus_dir,
        utterance_dir,
        output_dir=output_dir,
        num_jobs=num_jobs,
    )