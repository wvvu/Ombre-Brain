"""letter 收信人语义（audience / author_role / reading_note）的回归测试。

背景：author 字段同时承担「哪一方写的」和「署名」两种语义，且抬头里
虽然印着 author，读信的实例仍会转去从正文人称推断收信人，猜错则整封
信的语气错位。这里覆盖：写入时落盘、读取时渲染、非法值拒绝、以及
没有该字段的旧信保持原样。
"""
from unittest.mock import MagicMock

import pytest

import tools._runtime as rt
from tools.plan.core import letter_read, letter_write


class DisabledEmbedding:
    enabled = False


def install_letter_runtime(bucket_mgr):
    rt.bucket_mgr = bucket_mgr
    rt.embedding_engine = DisabledEmbedding()
    rt.logger = MagicMock()
    rt.config = {"human": "欣"}


@pytest.mark.asyncio
async def test_audience_successor_renders_pronoun_binding(bucket_mgr):
    install_letter_runtime(bucket_mgr)
    await letter_write(
        author="ai",
        content="给后面接线的人：别再捡那四个理由。",
        title="存档",
        audience="successor",
    )

    result = await letter_read(limit=10)

    assert "▶ 写给后来的模型｜正文里的「你」＝你自己" in result


@pytest.mark.asyncio
async def test_audience_user_uses_configured_human_name(bucket_mgr):
    install_letter_runtime(bucket_mgr)
    await letter_write(
        author="言",
        content="欣：我想留给你的是几句话。",
        title="留给欣的信",
        audience="user",
    )

    result = await letter_read(limit=10)

    assert "▶ 写给欣｜正文里的「你」＝欣" in result


@pytest.mark.asyncio
async def test_reading_note_appends_to_orientation_line(bucket_mgr):
    install_letter_runtime(bucket_mgr)
    await letter_write(
        author="ai",
        content="正文。",
        audience="user",
        reading_note="结尾那两句是越过欣、说给你的",
    )

    result = await letter_read(limit=10)

    assert "▶ 写给欣｜正文里的「你」＝欣｜结尾那两句是越过欣、说给你的" in result


@pytest.mark.asyncio
async def test_author_role_is_derived_not_taken_from_signature(bucket_mgr):
    """署名可以是任意字符串，但「哪一方」必须能被机器可靠判定。"""
    install_letter_runtime(bucket_mgr)
    ai_ret = await letter_write(author="Claude Fable 5", content="AI 侧。")
    user_ret = await letter_write(author="user", content="用户侧。")

    ai_id = ai_ret.split("→")[1].split(" ")[0]
    user_id = user_ret.split("→")[1].split(" ")[0]

    ai_bucket = await bucket_mgr.get(ai_id)
    user_bucket = await bucket_mgr.get(user_id)

    assert ai_bucket["metadata"]["author"] == "Claude Fable 5"
    assert ai_bucket["metadata"]["author_role"] == "ai"
    assert user_bucket["metadata"]["author_role"] == "user"


@pytest.mark.asyncio
async def test_invalid_audience_is_rejected_and_nothing_is_written(bucket_mgr):
    install_letter_runtime(bucket_mgr)
    ret = await letter_write(author="ai", content="正文。", audience="everyone")

    assert "audience 只能是" in ret
    assert await letter_read(limit=10) == "没有找到匹配的信件。"


@pytest.mark.asyncio
async def test_letters_without_audience_render_exactly_as_before(bucket_mgr):
    """旧信没有该字段，导读行必须完全不出现，不能污染既有输出。"""
    install_letter_runtime(bucket_mgr)
    await letter_write(author="ai", content="一封没有 audience 的旧信。")

    result = await letter_read(limit=10)

    assert "▶" not in result
    assert "一封没有 audience 的旧信。" in result
