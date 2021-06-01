import asyncio
import contextlib
import sys

from functools import wraps
from pathlib import Path
from typing import AsyncContextManager, Dict, List, Optional, cast
from unittest import mock

import aiohttp
import click
import git
import pytest
import tmt
import yarl

from click.testing import CliRunner
from envparse import env
from giturlparse import parse as parse_git_url
from loguru import logger


assert sys.version_info > (3, 7), "Script can't be run with lower version of python"

REPO_ROOT = Path(__file__).parents[1]

# Mapping of tmt plans origin_vm_name provisioner definition to tft composes
VM2COMPOSES = {
    "c2r_centos7_template": "CentOS-7",
    "c2r_centos8_template": "CentOS-8",
    # TODO is not yet enabled in TFT https://gitlab.com/testing-farm/general/-/issues/32
    # "c2r_oracle7_template": "OracleLinux-7",
    # "c2r_oracle8_template": "OracleLinux-8",
}


def get_compose_from_provision_data(data: List[dict]) -> str:
    """Get compose name from the provisioning metadata of the libvirt provisoner.

    For local development a libvirt provisioner is used, however, TFT
    is replacing the provisioner with its own type, requiring to specify the
    compose type (OS name). This function computes the compose name from
    local metadata.
    """
    assert len(data) == 1, "Expecting only one dict with provisioning data."
    provision_data = data[0]
    assert provision_data["how"] == "libvirt", "Expecting here only libvirt provisioner."
    try:
        return VM2COMPOSES[provision_data["origin_vm_name"]]
    except KeyError:
        logger.critical(f"VM name {provision_data['origin_vm_name']} is not registered in VM2COMPOSES variable.")
        raise


class TFTHelper:
    """General class to intereact with the TFT."""

    def __init__(
        self,
        web_client: "WebClient",
        plans: List[str],
        remote: str = "origin",
    ):
        # tft/tmt related
        self.web_client: "WebClient" = web_client
        self._api_url = yarl.URL(env.str("TFT_SERVICE_URL")) / f"v{env.str('TFT_API_VERSION')}"
        self._api_request_url: yarl.URL = self._api_url / "requests"
        self.jobs_registry: Dict[str, asyncio.Task] = {}
        self.plans: List[str] = plans

        # git repo related
        self.repo: git.Repo = git.Repo()
        self.remote: str = remote
        self.repo_url: str = parse_git_url(next(self.repo.remote(remote).urls)).url2https
        self.commit_hexsha: str = self.repo.commit().hexsha

    async def tft_health_check(self):
        # TODO cover by tests
        logger.debug("Verifying available composes...")
        async with self.web_client.session.get(url=self._api_url / "composes") as composes:
            composes = await composes.json()
            assert set(compose["name"] for compose in composes["composes"]).issuperset(set(VM2COMPOSES.values())), (
                f"Unknown composes specified in VM2COMPOSES. "
                f"Check for available composes at {self._api_url / 'composes'}."
            )
        logger.debug("PASSED")
        logger.info("TFT api health check PASSED")

    async def repo_health_check(self):
        # TODO cover by tests
        logger.debug("Verifying repo not commited changes...")
        if self.repo.index.diff(None) or self.repo.index.diff("HEAD"):
            logger.warning(f"Some files contains not commited changes.")

        logger.debug("Verifying repo synced with the remote...")
        self.repo.remote("origin").fetch()
        try:
            # Check if there is at least one commit ahead in local branch
            next(
                self.repo.iter_commits(
                    f"{self.remote}/{self.repo.active_branch.name}.." f"{self.repo.active_branch.name}",
                )
            )
        except StopIteration:
            # this means local and remote are in sync (no commits ahead)
            pass
        else:
            raise git.GitError(
                f"Local branch {repr(self.repo.active_branch.name)} "
                f"is ahead of the remote. Changes needs to be pushed."
            )
        logger.info("Repo state check PASSED")

    async def run_tmt_tests(self):
        """Running tft given tmt tests."""
        logger.info("Verifying TFT api health...")
        await self.tft_health_check()

        logger.info("Verifying repo state...")
        await self.repo_health_check()

        logger.info("Submitting tmt tests to tft runner...")
        for plan in tmt.Tree(".").plans(names=self.plans):
            tft_request_id = await self._submit_plan(plan)
            self.jobs_registry[plan.name] = asyncio.create_task(
                self._watch_test_id(tft_request_id, plan.name),
                name=plan.name,
            )
        try:
            await asyncio.gather(*self.jobs_registry.values())
        # if API returns non 200 codes, we're cancelling the corresponding task
        #   so we're handling this case
        except asyncio.exceptions.CancelledError:
            logger.warning("Some of plans were unable to be submitted or finished. Check previous log entries.")

    async def _submit_plan(self, plan: tmt.Plan):
        async with self.web_client.session.post(
            url=self._api_request_url,
            json={
                "api_key": env.str("TFT_API_KEY"),
                "test": {
                    "fmf": {
                        "url": self.repo_url,
                        "ref": self.commit_hexsha,
                        "name": plan.name,
                    },
                },
                "environments": [
                    {
                        "arch": "x86_64",
                        "os": {
                            "compose": get_compose_from_provision_data(plan.provision.data),
                        },
                        "variables": plan.environment,
                    },
                ],
            },
        ) as resp:
            logger.info(f"Plan {plan.name} submitted to TFT")
            if resp.status == 200:
                data = await resp.json()
            else:
                self.jobs_registry[plan.name].cancel()
                logger.warning(f"Plan {plan} failed to be submitted. Response status is: {resp.status}")
                logger.debug(repr(resp))
                return
            return data["id"]

    async def _watch_test_id(self, test_id: str, plan: str):
        while True:
            async with self.web_client.session.get(self._api_request_url / test_id) as resp:
                if resp.status == 200:
                    data = await resp.json()
                else:
                    self.jobs_registry[plan].cancel()
                    logger.warning(f"Plan {plan} failed to fetch state. Response status is: {resp.status}")
                    logger.debug(repr(resp))
                    return
                logger.debug(f"{plan} \t {test_id} \t {data['state']}")
                if data["state"] in ("complete", "error"):
                    self.jobs_registry[plan].done()
                    logger.info(f"Plan {plan} finished with status: {data['state']}")
                    logger.debug(
                        f"Results available at:\n"
                        f"http://artifacts.dev.testing-farm.io/{test_id}/\n"
                        f"http://artifacts.dev.testing-farm.io/{test_id}/pipeline.log\n"
                    )
                    return
            await asyncio.sleep(3)


class WebClient:
    """Simple web client.

    Takes care to create only single session per one command run.
    Provides teardown mechanism.
    """

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self._loop = asyncio.get_event_loop()

    def prepare(self):
        self.session = aiohttp.ClientSession()
        logger.debug("Web client prepared")

    async def destroy(self):
        await self.session.close()
        logger.debug("Web client destroyed")


def build_client() -> WebClient:
    client = WebClient()
    logger.info("Web client was built")
    return client


def coro(f):
    """Simple hack for enabling coroutines as a click commands.

    More info:
        https://github.com/pallets/click/issues/85#issuecomment-503464628
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper


@click.command()
@click.option(
    "-p",
    "--plans",
    default=["/plans/"],
    multiple=True,
    show_default=True,
    help="Plan names. i.e. -p /plan/name. Could be multiple",
)
@click.option(
    "-r",
    "--remote-name",
    default="origin",
    show_default=True,
    help=(
        "Git remote name from which the content of the repo will be cloned at "
        "current commit. Warning: changes should be pushed to the remote "
        "before running this script."
    ),
)
@click.option("-v", "--verbose", count=True)
@coro
async def cli(
    plans: List[str],
    remote_name: str,
    verbose: int,
) -> None:
    # some housekeeping
    env.read_envfile(".env")
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if env.bool("DEBUG") or verbose > 0 else "INFO")

    # submit tmt plans for execution on tft
    client = build_client()
    client.prepare()
    tft = TFTHelper(
        client,
        plans=[plan.strip() for plan in plans],
        remote=remote_name.strip(),
    )
    await tft.run_tmt_tests()
    await client.destroy()


# ----------------UNIT TESTS ------------------------ #
post_reps_stub = {
    "id": "7156aa79-88c0-4079-aed1-1ede7491d77d",
    "test": {
        "fmf": {
            "url": "https://github.com/ZhukovGreen/convert2rhel.git",
            "ref": "0c3d0608d452088e89f10fa73ea993d0cc5dd4de",
            "path": ".",
            "name": "/plans/integration/inhibit-if-kmods-is-not-supported/vm_centos7/bad",
        },
        "script": None,
        "sti": None,
    },
    "state": "new",
    "environments": [
        {
            "arch": "x86_64",
            "os": {"compose": "CentOS-7"},
            "pool": None,
            "variables": {
                "DEBUG": "False",
                "RHSM_POOL": "secret",
                "RHSM_PASSWORD": "secret",
                "RHSM_USERNAME": "secret",
                "RHSM_SERVER_URL": "secret",
                "ANSIBLE_BUILD_RPM": "False",
                "ANSIBLE_RPM_PROVIDER": "url",
                "ANSIBLE_RPM_URL_EL7": "secret",
                "ANSIBLE_RPM_URL_EL8": "secret",
                "ANSIBLE_REPO_ROOT": "",
                "TFT_SERVICE_URL": "secret",
                "TFT_API_VERSION": "secret",
                "TFT_API_KEY": "secret",
            },
            "artifacts": None,
            "settings": None,
            "tmt": None,
        }
    ],
    "notification": None,
    "created": "None",
    "updated": "None",
}
get_resp_stub = {
    "id": "7156aa79-88c0-4079-aed1-1ede7491d77d",
    "user_id": "user",
    "test": {
        "fmf": {
            "name": "/plans/integration/inhibit-if-kmods-is-not-supported/vm_centos7/bad",
            "path": ".",
            "ref": "0c3d0608d452088e89f10fa73ea993d0cc5dd4de",
            "url": "https://github.com/ZhukovGreen/convert2rhel.git",
        },
        "script": None,
        "sti": None,
    },
    "state": "new",
    "environments_requested": [
        {
            "arch": "x86_64",
            "artifacts": None,
            "os": {"compose": "CentOS-7"},
            "pool": None,
            "settings": None,
            "tmt": None,
            "variables": {
                "ANSIBLE_BUILD_RPM": "False",
                "ANSIBLE_REPO_ROOT": "",
                "ANSIBLE_RPM_PROVIDER": "url",
                "ANSIBLE_RPM_URL_EL7": "secret",
                "ANSIBLE_RPM_URL_EL8": "secret",
                "DEBUG": "False",
                "RHSM_PASSWORD": "secret",
                "RHSM_POOL": "secret",
                "RHSM_SERVER_URL": "secret.rhsm.stage.redhat.com",
                "RHSM_USERNAME": "secret",
                "TFT_API_KEY": "secret",
                "TFT_API_VERSION": "secret",
                "TFT_SERVICE_URL": "secret",
            },
        }
    ],
    "notes": None,
    "result": None,
    "run": None,
    "created": "2021-06-10 16:12:08.919837",
    "updated": "2021-06-10 16:12:08.919849",
}


class FakePostResponse:
    status = 200

    def __init__(self, json_stub: dict, counter: int = 0, update_stub: dict = None):
        self.json_stub = json_stub
        self._counter = counter
        self._update_stub = update_stub or {}

    async def json(self):
        self._counter -= 1
        if self._counter == 0:
            self.json_stub.update(self._update_stub)
        return self.json_stub


def response_factory(response: FakePostResponse) -> AsyncContextManager[FakePostResponse]:
    @contextlib.asynccontextmanager
    async def fake_post_response(*args, **kwargs) -> FakePostResponse:
        try:
            yield response
        finally:
            pass

    return cast(AsyncContextManager[FakePostResponse], fake_post_response)


@pytest.fixture(autouse=True)
def disable_removing_logger(monkeypatch):
    monkeypatch.setattr(logger, "remove", mock.Mock())


@pytest.mark.parametrize(
    ("response_status", "exit_code"),
    (
        (200, 0),
        (404, 1),
    ),
)
def test_cli_main(monkeypatch, response_status, exit_code):
    FakePostResponse.status = response_status
    get_response = FakePostResponse(
        get_resp_stub,
        counter=3,
        update_stub={"state": "complete"},
    )
    post_response = FakePostResponse(post_reps_stub)

    monkeypatch.setattr(aiohttp.ClientSession, "post", response_factory(post_response))
    monkeypatch.setattr(aiohttp.ClientSession, "get", response_factory(get_response))
    monkeypatch.setattr(asyncio, "sleep", mock.AsyncMock())

    monkeypatch.setattr(TFTHelper, "tft_health_check", mock.AsyncMock())
    monkeypatch.setattr(TFTHelper, "repo_health_check", mock.AsyncMock())
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "-p /plans/integration/inhibit-if-kmods-is-not-supported/vm_centos",
            "-r origin",
            "-v",
        ],
    )
    assert result.exit_code == exit_code


def test_bad_vm_origin(monkeypatch):
    global VM2COMPOSES
    VM2COMPOSES = {}
    monkeypatch.setattr(TFTHelper, "tft_health_check", mock.AsyncMock())
    monkeypatch.setattr(TFTHelper, "repo_health_check", mock.AsyncMock())
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "-p /plans/integration/inhibit-if-kmods-is-not-supported/vm_centos",
            "-r origin",
            "-v",
        ],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, KeyError)


# -------------------------------------------------- #

__name__ == "__main__" and cli()
