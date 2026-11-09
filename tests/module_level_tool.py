"""A tool kept at module level, the way an application that names its tools by module would keep it."""

from strands_acp_client import make_acp_agent
from tests.conftest import FAKE_ARGV

acp_agent = make_acp_agent(agents={"fake": FAKE_ARGV}, timeout=10)
