"""Controla a gravação de episódios do rosetta pelos botões do controle esquerdo.

    X (button_lower)              : inicia a gravação
    Y (button_upper)              : para e salva o episódio
    Gatilho do indicador (index)  : para e deleta o episódio em andamento

O gatilho lateral (press_middle) e o analógico ficam livres para a base.

O episode_recorder é a fonte da verdade do estado: cada botão só chama o
serviço correspondente, e o recorder recusa o que não faz sentido (iniciar já
gravando, salvar sem gravação). Deletar só age sobre uma gravação em andamento,
para um toque sem querer nunca apagar um episódio já salvo.

O prompt da tarefa é o default_prompt do episode_recorder (definido na etapa
Recorder do painel ou no episode_recorder.yaml).

Cada transição é publicada em ~/state (JSON, latched), para monitoramento:

    ros2 topic echo /record_buttons/state --qos-durability transient_local
"""

import json
import threading
import time

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String
from std_srvs.srv import Trigger

from quest2ros.msg import OVR2ROSInputs
from rosetta_interfaces.srv import StartRecording

SERVICE_TIMEOUT_S = 3.0
DELETE_RETRY_S = 0.25
DELETE_MAX_WAIT_S = 10.0
TRIGGER_PRESSED = 0.5  # gatilhos são analógicos (0.0 a 1.0)


class RecordButtons(Node):
    def __init__(self):
        super().__init__('record_buttons')

        inputs_topic = self.declare_parameter('inputs_topic', '/q2r_left_hand_inputs').value
        recorder_ns = self.declare_parameter('recorder_ns', '/episode_recorder').value

        cbg = ReentrantCallbackGroup()
        self.start_client = self.create_client(
            StartRecording, f'{recorder_ns}/start_recording', callback_group=cbg)
        self.cancel_client = self.create_client(
            Trigger, f'{recorder_ns}/cancel_recording', callback_group=cbg)
        self.delete_client = self.create_client(
            Trigger, f'{recorder_ns}/delete_last_bag', callback_group=cbg)

        self.state_pub = self.create_publisher(
            String, '~/state',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

        self.create_subscription(OVR2ROSInputs, inputs_topic, self._on_inputs, 10, callback_group=cbg)

        self.lock = threading.Lock()  # um comando por vez: ignora toques enquanto o anterior roda
        self.prev = {'start': False, 'save': False, 'delete': False}
        self.recording = False
        self.started_at = None
        self.saved = 0
        self.deleted = 0
        self._publish_state('pronto')

        self.get_logger().info(
            f'Gravação pelo Quest ({inputs_topic}): X inicia, Y salva, gatilho do indicador deleta.')

    # ── Botões ────────────────────────────────────────────────────────────
    def _on_inputs(self, msg: OVR2ROSInputs):
        pressed = {
            'start': msg.button_lower,
            'save': msg.button_upper,
            'delete': msg.press_index > TRIGGER_PRESSED,
        }
        commands = {'start': self.start, 'save': self.save, 'delete': self.delete}
        for name, is_down in pressed.items():
            if is_down and not self.prev[name]:
                self._run(commands[name])
            self.prev[name] = is_down

    def _run(self, command):
        if not self.lock.acquire(blocking=False):
            self.get_logger().warn('Comando anterior ainda em andamento; toque ignorado.')
            return
        try:
            command()
        finally:
            self.lock.release()

    # ── Comandos ──────────────────────────────────────────────────────────
    def start(self):
        # Prompt vazio: o recorder usa o default_prompt dele
        started = self._call(self.start_client, StartRecording.Request(prompt=''))
        if started is None:
            return self._fail('episode_recorder não respondeu; a etapa Recorder está habilitada?')
        if not started.accepted:
            # O caso comum é apertar X já gravando
            self.get_logger().warn(f'Gravação não iniciada: {started.message}')
            return self._publish_state('aviso', started.message)
        self.recording = True
        self.started_at = time.time()
        self.get_logger().info('Gravando...')
        self._publish_state('gravando')

    def save(self):
        stopped = self._call(self.cancel_client, Trigger.Request())
        if stopped is None:
            return self._fail('episode_recorder não respondeu; a etapa Recorder está habilitada?')
        if not stopped.success:
            self.get_logger().info('Nada gravando; nada a salvar.')
            return self._publish_state('aviso', 'nada gravando')
        self.recording = False
        self.saved += 1
        self.get_logger().info(f'Episódio salvo ({self.saved} nesta sessão).')
        self._publish_state('salvo')

    def delete(self):
        stopped = self._call(self.cancel_client, Trigger.Request())
        if stopped is None:
            return self._fail('episode_recorder não respondeu; a etapa Recorder está habilitada?')
        if not stopped.success:
            self.get_logger().info('Nada gravando; nada a deletar.')
            return self._publish_state('aviso', 'nada gravando')

        self.recording = False
        # O bag só fica apagável depois que o writer fecha
        deadline = time.monotonic() + DELETE_MAX_WAIT_S
        while time.monotonic() < deadline:
            deleted = self._call(self.delete_client, Trigger.Request())
            if deleted is not None and deleted.success:
                self.deleted += 1
                self.get_logger().info(f'Episódio deletado ({deleted.message}).')
                return self._publish_state('deletado')
            if deleted is not None and 'recording in progress' not in deleted.message:
                return self._fail(f'não consegui apagar o bag: {deleted.message}')
            time.sleep(DELETE_RETRY_S)
        self._fail('tempo esgotado esperando o bag fechar para apagar')

    # ── Auxiliares ────────────────────────────────────────────────────────
    def _call(self, client, request):
        if not client.wait_for_service(timeout_sec=SERVICE_TIMEOUT_S):
            return None
        future = client.call_async(request)
        deadline = time.monotonic() + SERVICE_TIMEOUT_S
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        return future.result() if future.done() else None

    def _fail(self, message):
        self.get_logger().error(message)
        self._publish_state('erro', message)

    def _publish_state(self, event, message=''):
        self.state_pub.publish(String(data=json.dumps({
            'event': event,
            'message': message,
            'recording': self.recording,
            'started_at': self.started_at if self.recording else None,
            'saved': self.saved,
            'deleted': self.deleted,
            'stamp': time.time(),
        })))


def main(args=None):
    rclpy.init(args=args)
    node = RecordButtons()
    # Multithread: os callbacks esperam respostas de serviço sem travar o executor
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
