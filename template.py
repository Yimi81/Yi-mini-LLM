from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Sequence, Tuple, Union, Literal
from loguru import logger
from transformers import PreTrainedTokenizer

from utils import (
    SLOTS,
    Formatter,
    Role,
    infer_max_len,
    EmptyFormatter,
    FunctionFormatter,
    StringFormatter,
    ToolFormatter,
)


@dataclass
class Template:
    format_user: "Formatter"
    format_assistant: "Formatter"
    format_system: "Formatter"
    format_function: "Formatter"
    format_observation: "Formatter"
    format_tools: "Formatter"
    format_separator: "Formatter"
    default_system: str
    stop_words: List[str]
    efficient_eos: bool
    replace_eos: bool
    force_system: bool

    def encode_oneturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        cutoff_len: int = 1_000_000,
        reserved_label_len: int = 1,
    ) -> Tuple[List[int], List[int]]:
        r"""
        Returns a single pair of token ids representing prompt and response respectively.
        """
        encoded_pairs = self._encode(
            tokenizer, messages, system, tools, cutoff_len, reserved_label_len
        )
        prompt_ids = []
        for query_ids, resp_ids in encoded_pairs[:-1]:
            prompt_ids += query_ids + resp_ids
        prompt_ids = prompt_ids + encoded_pairs[-1][0]
        answer_ids = encoded_pairs[-1][1]
        return prompt_ids, answer_ids

    def encode_multiturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        cutoff_len: int = 1_000_000,
        reserved_label_len: int = 1,
    ) -> Sequence[Tuple[List[int], List[int]]]:
        r"""
        Returns multiple pairs of token ids representing prompts and responses respectively.
        """
        return self._encode(
            tokenizer, messages, system, tools, cutoff_len, reserved_label_len
        )

    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: List[Dict[str, str]],
        system: str,
        tools: str,
        cutoff_len: int,
        reserved_label_len: int,
    ) -> Sequence[Tuple[List[int], List[int]]]:
        r"""
        Encodes formatted inputs to pairs of token ids.
        Turn 0: system + query        resp
        Turn t: sep + query           resp
        """
        system = system or self.default_system
        encoded_messages = []
        for i, message in enumerate(messages):
            elements = []
            if i == 0 and (system or tools or self.force_system):
                tool_text = self.format_tools.apply(content=tools)[0] if tools else ""
                elements += self.format_system.apply(content=(system + tool_text))
            elif i > 0 and i % 2 == 0:
                elements += self.format_separator.apply()

            if message["role"] == Role.USER.value:
                elements += self.format_user.apply(
                    content=message["content"], idx=str(i // 2)
                )
            elif message["role"] == Role.ASSISTANT.value:
                elements += self.format_assistant.apply(content=message["content"])
            elif message["role"] == Role.OBSERVATION.value:
                elements += self.format_observation.apply(content=message["content"])
            elif message["role"] == Role.FUNCTION.value:
                elements += self.format_function.apply(content=message["content"])
            else:
                raise NotImplementedError("Unexpected role: {}".format(message["role"]))

            encoded_messages.append(self._convert_elements_to_ids(tokenizer, elements))

        return self._make_pairs(encoded_messages, cutoff_len, reserved_label_len)

    def _convert_elements_to_ids(
        self,
        tokenizer: "PreTrainedTokenizer",
        elements: List[Union[str, Dict[str, str]]],
    ) -> List[int]:
        r"""
        Converts elements to token ids.
        """
        token_ids = []
        for elem in elements:
            if isinstance(elem, str):
                if len(elem) != 0:
                    token_ids += tokenizer.encode(elem, add_special_tokens=False)
            elif isinstance(elem, dict):
                token_ids += [tokenizer.convert_tokens_to_ids(elem.get("token"))]
            elif isinstance(elem, set):
                if "bos_token" in elem and tokenizer.bos_token_id is not None:
                    token_ids += [tokenizer.bos_token_id]
                elif "eos_token" in elem and tokenizer.eos_token_id is not None:
                    token_ids += [tokenizer.eos_token_id]
            else:
                raise ValueError(
                    "Input must be string, set[str] or dict[str, str], got {}".format(
                        type(elem)
                    )
                )

        return token_ids

    def _make_pairs(
        self,
        encoded_messages: Sequence[List[int]],
        cutoff_len: int,
        reserved_label_len: int,
    ) -> Sequence[Tuple[List[int], List[int]]]:
        encoded_pairs = []
        total_length = 0
        for i in range(0, len(encoded_messages), 2):
            if total_length >= cutoff_len:
                break

            max_source_len, max_target_len = infer_max_len(
                source_len=len(encoded_messages[i]),
                target_len=len(encoded_messages[i + 1]),
                max_len=(cutoff_len - total_length),
                reserved_label_len=reserved_label_len,
            )
            source_ids = encoded_messages[i][:max_source_len]
            target_ids = encoded_messages[i + 1][:max_target_len]
            total_length += len(source_ids) + len(target_ids)
            encoded_pairs.append((source_ids, target_ids))

        return encoded_pairs


templates: Dict[str, Template] = {}


def _register_template(
    name: str,
    format_user: Optional["Formatter"] = None,
    format_assistant: Optional["Formatter"] = None,
    format_system: Optional["Formatter"] = None,
    format_function: Optional["Formatter"] = None,
    format_observation: Optional["Formatter"] = None,
    format_tools: Optional["Formatter"] = None,
    format_separator: Optional["Formatter"] = None,
    default_system: str = "",
    stop_words: List[str] = [],
    efficient_eos: bool = False,
    replace_eos: bool = False,
    force_system: bool = False,
) -> None:
    r"""
    Registers a chat template.

    To add the following chat template:
    ```
    [HUMAN]:
    user prompt here
    [AI]:
    model response here

    [HUMAN]:
    user prompt here
    [AI]:
    model response here
    ```

    The corresponding code should be:
    ```
    _register_template(
        name="custom",
        format_user=StringFormatter(slots=["[HUMAN]:\n{{content}}\n[AI]:\n"]),
        format_separator=EmptyFormatter(slots=["\n\n"]),
        efficient_eos=True,
    )
    ```
    """
    eos_slots = [] if efficient_eos else [{"eos_token"}]
    template_class = Template
    default_user_formatter = StringFormatter(slots=["{{content}}"])
    default_assistant_formatter = StringFormatter(slots=["{{content}}"] + eos_slots)
    default_function_formatter = FunctionFormatter(
        slots=["Action: {{name}}\nAction Input: {{arguments}}"] + eos_slots
    )
    default_tool_formatter = ToolFormatter(tool_format="default")
    default_separator_formatter = EmptyFormatter()
    templates[name] = template_class(
        format_user=format_user or default_user_formatter,
        format_assistant=format_assistant or default_assistant_formatter,
        format_system=format_system or default_user_formatter,
        format_function=format_function or default_function_formatter,
        format_observation=format_observation or format_user or default_user_formatter,
        format_tools=format_tools or default_tool_formatter,
        format_separator=format_separator or default_separator_formatter,
        default_system=default_system,
        stop_words=stop_words,
        efficient_eos=efficient_eos,
        replace_eos=replace_eos,
        force_system=force_system,
    )


def _add_or_replace_eos_token(tokenizer: "PreTrainedTokenizer", eos_token: str) -> None:
    is_added = tokenizer.eos_token_id is None
    num_added_tokens = tokenizer.add_special_tokens({"eos_token": eos_token})

    if is_added:
        logger.info("Add eos token: {}".format(tokenizer.eos_token))
    else:
        logger.info("Replace eos token: {}".format(tokenizer.eos_token))

    if num_added_tokens > 0:
        logger.warning("New tokens have been added, make sure `resize_vocab` is True.")


def _jinja_escape(content: str) -> str:
    return content.replace("\n", r"\n").replace("'", r"\'")


def _convert_slots_to_jinja(slots: "SLOTS", tokenizer: "PreTrainedTokenizer", placeholder: str = "content") -> str:
    slot_items = []
    for slot in slots:
        if isinstance(slot, str):
            slot_pieces = slot.split("{{content}}")
            if slot_pieces[0]:
                slot_items.append("'" + _jinja_escape(slot_pieces[0]) + "'")
            if len(slot_pieces) > 1:
                slot_items.append(placeholder)
                if slot_pieces[1]:
                    slot_items.append("'" + _jinja_escape(slot_pieces[1]) + "'")
        elif isinstance(slot, set):
            if "bos_token" in slot:
                slot_items.append("'" + tokenizer.bos_token + "'")
            elif "eos_token" in slot:  # do not use {{ eos_token }} since it may be replaced
                slot_items.append("'" + tokenizer.eos_token + "'")
        elif isinstance(slot, dict):
            raise ValueError("Dict is not supported.")

    return " + ".join(slot_items)


def _get_jinja_template(template: "Template", tokenizer: "PreTrainedTokenizer") -> str:
    jinja_template = ""

    if template.default_system:
        jinja_template += "{% set system_message = '" + _jinja_escape(template.default_system) + "' %}"

    jinja_template += (
        "{% if messages[0]['role'] == 'system' %}" "{% set system_message = messages[0]['content'] %}" "{% endif %}"
    )

    system_message = _convert_slots_to_jinja(template.format_system.apply(), tokenizer, placeholder="system_message")
    if template.force_system:
        jinja_template += "{{ " + system_message + " }}"
    else:
        jinja_template += "{% if system_message is defined %}{{ " + system_message + " }}{% endif %}"

    jinja_template += "{% for message in messages %}"
    jinja_template += "{% set content = message['content'] %}"
    jinja_template += "{% if message['role'] == 'user' %}"
    user_message = _convert_slots_to_jinja(template.format_user.apply(), tokenizer)
    jinja_template += "{{ " + user_message + " }}"
    jinja_template += "{% elif message['role'] == 'assistant' %}"
    assistant_message = _convert_slots_to_jinja(
        template.format_assistant.apply() + template.format_separator.apply(), tokenizer
    )
    jinja_template += "{{ " + assistant_message + " }}"
    jinja_template += "{% endif %}"
    jinja_template += "{% endfor %}"
    return jinja_template


def get_template_and_fix_tokenizer(
    tokenizer: "PreTrainedTokenizer",
    name: Optional[str] = None,
) -> Template:
    if name is None:
        template = templates["empty"]  # placeholder
    else:
        template = templates.get(name, None)
        if template is None:
            raise ValueError("Template {} does not exist.".format(name))

    stop_words = template.stop_words
    if template.replace_eos:
        if not stop_words:
            raise ValueError("Stop words are required to replace the EOS token.")

        _add_or_replace_eos_token(tokenizer, eos_token=stop_words[0])
        stop_words = stop_words[1:]

    if tokenizer.eos_token_id is None:
        _add_or_replace_eos_token(tokenizer, eos_token="<|endoftext|>")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Add pad token: {}".format(tokenizer.pad_token))

    if stop_words:
        num_added_tokens = tokenizer.add_special_tokens(
            dict(additional_special_tokens=stop_words), replace_additional_special_tokens=False
        )
        logger.info("Add {} to stop words.".format(",".join(stop_words)))
        if num_added_tokens > 0:
            logger.warning("New tokens have been added, make sure `resize_vocab` is True.")

    try:
        tokenizer.chat_template = _get_jinja_template(template, tokenizer)
    except ValueError:
        logger.info("Cannot add this chat template to tokenizer.")

    return template


_register_template(
    name="default",
    format_user=StringFormatter(slots=["Human: {{content}}\nAssistant: "]),
    format_system=StringFormatter(slots=["{{content}}\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
)


_register_template(
    name="empty",
    format_user=StringFormatter(slots=["{{content}}"]),
    format_assistant=StringFormatter(slots=["{{content}}"]),
)


_register_template(
    name="yi",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    default_system="You are a helpful assistant.",
    stop_words=["<|im_end|>"],
    replace_eos=True,
)
