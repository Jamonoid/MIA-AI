"""
Tests para conversations/message_handler.py

Verifica el patrón request-response asyncio.Event:
- wait + receive → desbloquea
- timeout → retorna None
- cleanup → desbloquea waiters pendientes
"""

import asyncio

import pytest

from mia.conversations.message_handler import MessageHandler


@pytest.fixture
def handler():
    """MessageHandler fresco para cada test (no usar el singleton)."""
    return MessageHandler()


@pytest.mark.asyncio
async def test_wait_and_receive(handler):
    """Un waiter se desbloquea cuando llega el mensaje esperado."""
    client = "test-client-1"

    async def _respond_later():
        await asyncio.sleep(0.05)
        handler.handle_message(
            client, {"type": "frontend-playback-complete", "extra": 42}
        )

    asyncio.create_task(_respond_later())

    result = await handler.wait_for_response(
        client, "frontend-playback-complete", timeout=2.0
    )

    assert result is not None
    assert result["type"] == "frontend-playback-complete"
    assert result["extra"] == 42
    # El waiter debe haberse limpiado
    assert handler.active_waiters == 0


@pytest.mark.asyncio
async def test_wait_with_request_id(handler):
    """Waiters con request_id diferente no se interfieren."""
    client = "test-client-2"

    async def _respond():
        await asyncio.sleep(0.05)
        handler.handle_message(
            client, {"type": "response", "request_id": "req-B", "data": "B"}
        )
        await asyncio.sleep(0.05)
        handler.handle_message(
            client, {"type": "response", "request_id": "req-A", "data": "A"}
        )

    asyncio.create_task(_respond())

    # Esperar req-A (que llega segundo)
    result = await handler.wait_for_response(
        client, "response", request_id="req-A", timeout=2.0
    )

    assert result is not None
    assert result["data"] == "A"


@pytest.mark.asyncio
async def test_timeout_returns_none(handler):
    """Si no llega respuesta, timeout retorna None."""
    result = await handler.wait_for_response(
        "ghost-client", "never-coming", timeout=0.1
    )
    assert result is None
    assert handler.active_waiters == 0


@pytest.mark.asyncio
async def test_cleanup_unblocks_waiters(handler):
    """cleanup_client desbloquea waiters pendientes (retornan None)."""
    client = "test-client-3"

    async def _wait():
        return await handler.wait_for_response(
            client, "something", timeout=5.0
        )

    task = asyncio.create_task(_wait())

    # Dar tiempo para que el waiter se registre
    await asyncio.sleep(0.05)
    assert handler.active_waiters == 1

    # Cleanup — debe desbloquear
    handler.cleanup_client(client)
    result = await asyncio.wait_for(task, timeout=1.0)

    # El waiter retorna None porque se desbloqueó por cleanup, no por respuesta
    assert result is None
    assert handler.active_waiters == 0


@pytest.mark.asyncio
async def test_handle_message_no_waiter(handler):
    """handle_message retorna False si no hay waiter esperando."""
    result = handler.handle_message(
        "nobody", {"type": "orphan-message"}
    )
    assert result is False


@pytest.mark.asyncio
async def test_handle_message_returns_true(handler):
    """handle_message retorna True cuando desbloquea un waiter."""
    client = "test-client-4"

    async def _wait():
        return await handler.wait_for_response(
            client, "ack", timeout=2.0
        )

    task = asyncio.create_task(_wait())
    await asyncio.sleep(0.05)

    matched = handler.handle_message(client, {"type": "ack"})
    assert matched is True

    result = await asyncio.wait_for(task, timeout=1.0)
    assert result is not None


@pytest.mark.asyncio
async def test_multiple_clients_independent(handler):
    """Eventos de un cliente no afectan a otro."""
    client_a = "client-A"
    client_b = "client-B"

    async def _wait_a():
        return await handler.wait_for_response(
            client_a, "ping", timeout=2.0
        )

    async def _wait_b():
        return await handler.wait_for_response(
            client_b, "ping", timeout=2.0
        )

    task_a = asyncio.create_task(_wait_a())
    task_b = asyncio.create_task(_wait_b())
    await asyncio.sleep(0.05)

    # Solo responder a A
    handler.handle_message(client_a, {"type": "ping", "who": "A"})
    result_a = await asyncio.wait_for(task_a, timeout=1.0)
    assert result_a["who"] == "A"

    # B sigue esperando
    assert not task_b.done()

    # Cleanup B
    handler.cleanup_client(client_b)
    result_b = await asyncio.wait_for(task_b, timeout=1.0)
    assert result_b is None
