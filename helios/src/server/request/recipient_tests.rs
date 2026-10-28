//! CPU request preparation with the canonical recipient-header checkpoint frontend.

use super::*;
use crate::frontend::{Asset, FrontendAssets};
use crate::output::tagged::OutputStart;
use crate::server::protocol::{Protocol, decode};
use serde_json::json;
use sha2::{Digest, Sha256};

#[path = "../../../tests/support/frontend.rs"]
mod support;

#[test]
#[ignore = "requires FLARE_MUSE_SPARK_SNAPSHOT checkpoint tokenizer"]
fn recipient_request_preparation_preserves_protocols_tools_and_continuation() {
    let snapshot = std::path::PathBuf::from(std::env::var_os("FLARE_MUSE_SPARK_SNAPSHOT").unwrap());
    let asset = |path: std::path::PathBuf| {
        let bytes = std::fs::read(&path).unwrap();
        Asset {
            path,
            range: None,
            sha256: format!("{:x}", Sha256::digest(bytes)),
        }
    };
    let frontend = Frontend::load(
        FrontendAssets {
            tokenizer: asset(snapshot.join("tokenizer.json")),
            tokens: crate::frontend::test_support::tokens(&asset(
                snapshot.join("tokenizer_config.json"),
            ))
            .unwrap(),
            chat_template: Some(asset(
                std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../ignition/recipes/muse_spark/assets/chat_template.jinja"),
            )),
        },
        json!({"add_generation_prompt":true,"current_date":"2026-10-27"})
            .as_object()
            .unwrap()
            .clone(),
    )
    .unwrap();
    let mut config = support::model_config(crate::model::testing::metadata());
    config.contract.vocabulary = frontend.tokenizer().get_vocab_size(true);
    config.contract.input_vocabulary = config.contract.vocabulary;
    config.contract.vision.max_images = 0;
    config.contract.frontend=serde_json::from_value(json!({
        "tools":{"kind":7,"wrapping_newlines":0,"tag_separator":""},
        "reasoning":{"kind":5,"open":"<|start|>assistant to=self<|message|>","close":"<|eom|>","wrapping_newlines":0,"terminal_markers":[]},
        "assistant_marker":"<|start|>assistant","message_end_markers":["<|eom|>","<|eot|>"],
        "rejected_tokens":[],"position_dimensions":0
    })).unwrap();
    config.eos_token_ids = vec![200001, 200008];
    let compiler = crate::grammar::GrammarCompiler::new(
        frontend.tokenizer(),
        config.contract.vocabulary,
        &config.eos_token_ids,
        &crate::model::reasoning_delimiter_tokens(&frontend, &config.contract.frontend).unwrap(),
    )
    .unwrap();
    let prepare = |protocol, mut body: Value| {
        body["model"] = json!(config.model_id);
        let decoded = decode(protocol, body, false)?;
        prepare_inference(
            &frontend,
            None,
            &compiler,
            &config,
            &decoded.request,
            0,
            false,
        )
    };
    for (protocol, body) in [
        (
            Protocol::Chat,
            json!({"messages":[{"role":"user","content":"Hello 世界"}]}),
        ),
        (
            Protocol::Messages,
            json!({"max_tokens":32,"messages":[{"role":"user","content":"Hello 世界"}]}),
        ),
        (Protocol::Responses, json!({"input":"Hello 世界"})),
    ] {
        let request = prepare(protocol, body).unwrap();
        assert!(request.rendered_prompt.ends_with("<|start|>assistant"));
        assert_eq!(
            request.output.unwrap().content.start,
            OutputStart::RecipientHeader
        );
        assert!(!request.prompt.tokens.is_empty());
    }
    let tools = json!([{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"query":{"type":"string"}}}}}]);
    for choice in [
        json!("none"),
        json!("auto"),
        json!("required"),
        json!({"type":"function","function":{"name":"lookup"}}),
    ] {
        let request=prepare(Protocol::Chat,json!({"messages":[{"role":"user","content":"Find it"}],"tools":tools,"tool_choice":choice,
            "response_format":{"type":"json_schema","json_schema":{"name":"result","schema":{"type":"object"}}}})).unwrap();
        assert!(request.prompt.grammar.is_some());
        assert_eq!(
            request.output.unwrap().content.tools.is_empty(),
            choice == "none"
        );
        if choice == "required" {
            assert!(
                request
                    .rendered_prompt
                    .contains("You must call at least one")
            );
        }
    }
    let history = json!([{"role":"user","content":"Find it"},{"role":"assistant","tool_calls":[{"id":"a","type":"function","function":{"name":"lookup","arguments":"{\"query\":\"世界\"}"}}]},
        {"role":"tool","tool_call_id":"a","content":"Found"},{"role":"user","content":"Explain"}]);
    let request = prepare(Protocol::Chat, json!({"messages":history,"tools":tools})).unwrap();
    assert!(
        request
            .rendered_prompt
            .contains("<atem:parameter name=\"query\">世界</atem:parameter>")
    );
    assert!(
        request
            .rendered_prompt
            .contains("<tool_output name=\"lookup\">\nFound\n</tool_output>")
    );
    assert!(request.prompt.message_boundaries.len() >= 4);
    for (message, start) in [
        (
            json!({"role":"assistant","content":"Partial café"}),
            OutputStart::Content,
        ),
        (
            json!({"role":"assistant","content":"","reasoning_content":"Plan café"}),
            OutputStart::Reasoning,
        ),
    ] {
        let field = if start == OutputStart::Reasoning {
            json!("reasoning_content")
        } else {
            json!(true)
        };
        let request = prepare(
            Protocol::Chat,
            json!({"messages":[{"role":"user","content":"Hello"},message],
            "chat_template_kwargs":{"continue_final_message":field,"add_generation_prompt":false}}),
        )
        .unwrap();
        assert_eq!(request.output.unwrap().content.start, start);
    }
    let image = prepare(
        Protocol::Chat,
        json!({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]}]}),
    );
    assert!(image.is_err());
}
