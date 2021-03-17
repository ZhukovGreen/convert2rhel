import itertools
import logging
import re
import subprocess


try:
    from subprocess import check_output  # pylint: disable=no-name-in-module
except ImportError:
    # python2.6 doesn't have this
    def check_output(command):
        return subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
        ).communicate()[0]


from convert2rhel.systeminfo import system_info
from convert2rhel.utils import run_subprocess


logger = logging.getLogger(__name__)

KERNEL_REPO_RE = re.compile(
    "kernel-.+:(?P<version>(\d+\.)*(\d+)-(\d+\.)*(\d+)).el.+"
)
KERNEL_REPO_VER_SPLIT_RE = re.compile("\D+")


def _get_kmod_comparison_key(path):
    """Create a comparison key from the kernel module abs path.

    Converts /lib/modules/5.8.0-7642-generic/kernel/lib/a.ko.xz ->
    kernel/lib/a.ko.xz

    Why:
        all kernel modules lives in /lib/modules/{some kernel release}
        if we want to make sure that the kernel package is really presented
        on RHEL, we need to compare the full path, but because kernel release
        might be different, we compare the relative paths after kernel release.
    """
    return "/".join(path.split("/")[4:])


def get_host_kmods():
    try:
        # TODO make it work with utils.run_subprocess (now fails)
        kmod_str = check_output(
            'find /lib/modules/"$(uname -r)" -name "*.ko.xz"',
            shell=True,
        ).decode()
        assert kmod_str
    except (subprocess.CalledProcessError, AssertionError):
        logger.critical("Can't get list of kernel modules.")
    else:
        return set(
            _get_kmod_comparison_key(path)
            for path in kmod_str.rstrip("\n").split()
        )


def _repos_version_key(pkg_name):
    try:
        rpm_version = KERNEL_REPO_RE.search(pkg_name).group("version")
    except AttributeError:
        raise NotImplementedError(
            "Unexpected package:\n%s\n is a source of kernel modules."
            % pkg_name
        )
    else:
        return tuple(map(int, KERNEL_REPO_VER_SPLIT_RE.split(rpm_version)))


def get_most_recent_unique_kernel_pkgs(pkgs):
    """Return the most recent versions of all kernel packages.

    When we do scanning of kernel modules provided by kernel packages,
    it is expensive to check each kernel pkg. Considering the fact,
    that each new kernel pkg do not deprecate kernel modules we select only
    the most recent ones.

    :type pkgs: Iterable[str]
    """

    pkgs_groups = itertools.groupby(
        pkgs, lambda pkg_name: pkg_name.split(":")[0]
    )
    return (
        max(distinct_kernel_pkgs[1], key=_repos_version_key)
        for distinct_kernel_pkgs in pkgs_groups
        if distinct_kernel_pkgs[0].startswith("kernel")
    )


def get_rhel_supported_kmods(_refresh_system_info=False):
    """Return set of target RHEL supported kernel modules."""
    if _refresh_system_info:
        system_info.resolve_system_info()
    repoquery_repoids_args = (
        " ".join(
            (
                "--repoid " + repoid
                for repoid in system_info.default_rhsm_repoids
            )
        ),
    )
    kmod_pkgs_bytes, _ = run_subprocess(
        (
            "repoquery "
            "--releasever={releasever} "
            "{repoids_args} "
            "-f /lib/modules"
        ).format(
            releasever=system_info.releasever,
            repoids_args=repoquery_repoids_args,
        )
    )
    kmod_pkgs = get_most_recent_unique_kernel_pkgs(
        kmod_pkgs_bytes.decode().rstrip("\n").split()
    )
    rhel_kmods = set()
    for kmod_pkg in kmod_pkgs:
        rhel_kmods_bytes, _ = run_subprocess(
            (
                "repoquery "
                "--releasever={releasever} "
                "{repoids_args} "
                "-l {pkg}"
            ).format(
                releasever=system_info.releasever,
                repoids_args=repoquery_repoids_args,
                pkg=kmod_pkg,
            )
        )
        rhel_kmods.update(
            set(
                _get_kmod_comparison_key(kmod_path)
                for kmod_path in filter(
                    lambda path: path.endswith("ko.xz"),
                    rhel_kmods_bytes.decode().rstrip("\n").split(),
                )
            )
        )
    return rhel_kmods


def ensure_compatibility_of_kmods(_refresh_system_info=False):
    """Ensure if the host kernel modules are compatible with RHEL."""
    host_kmods = get_host_kmods()
    rhel_supported_kmods = get_rhel_supported_kmods(_refresh_system_info)
    if not host_kmods.issubset(rhel_supported_kmods):
        logger.critical(
            "The following kernel modules are not supported "
            "in RHEL:\n %s" % "\n".join(host_kmods - rhel_supported_kmods)
        )
    else:
        logger.debug("Kernel modules are compatible.")


def pre_ponr():
    ensure_compatibility_of_kmods()
