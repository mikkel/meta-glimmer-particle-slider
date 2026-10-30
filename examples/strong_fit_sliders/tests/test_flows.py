"""Synthetic CPU fixtures. No generated Glimmer evidence or paid judgments."""
import copy
import json
import math
import sys
import types
import unittest
from unittest.mock import patch
from strong_fit_sliders import flow as f
from strong_fit_sliders import evaluation as e


def candidate(config, case, text=None, n=0, **changes):
    text = case["reference"] if text is None else text
    ids = list(text.encode())  # Explicit fake tokenizer; never eligible for native handoff.
    c = dict(id=f"fixture-{case['id']}-{n}", case_id=case["id"],case_sha256=f.digest(case),
        config_sha256=f.digest(config),base_revision=config["base_revision"],tokenizer_sha256="f"*64,
        text=text,token_ids=ids,text_token_sha256=f.digest(dict(text=text,token_ids=ids)),
        sampling=config["selection_sampling"],termination="eos",origin="synthetic_fixture",
        condition=dict(system=config["positive_definition"],strength=0.,artifact_sha256=None),
        seed=config["selection_sampling"]["seeds"][n%4])
    c.update(changes)
    return c


def decision(action="answer", schema=None, **choices):
    schema = schema or f.questions()
    answers = {}
    for name, q in schema.items():
        if q["type"] == "choice":
            value = choices.get(name, {"relevance":"direct","adequacy":"complete","action":action}[name])
            answers[name]=dict(type="choice",choice=value,confidence=1.,
                               probabilities={k:float(k==value) for k in q["criteria"]})
        else:
            answers[name]=dict(type="score",score=3.,confidence=1.,probabilities={str(i):float(i==3) for i in range(4)})
    return dict(model=f.MODEL,answers=answers,usage=dict(input_tokens=1200,output_tokens=100))


def receipt(payload, response):
    return {"request_sha256": f.digest(payload), "response": response}


class Flows(unittest.TestCase):
    def setUp(self):
        self.config, self.cases = f.load("calibration")

    def test_frozen_splits_and_references(self):
        for name in ("calibration","constraint-precision","independent-judgment"):
            config, cases=f.load(name)
            self.assertEqual(len(cases),18)
            self.assertEqual([len(config["split_ids"][s]) for s in ("training","development","heldout")],[10,4,4])
        bad=copy.deepcopy(self.cases)
        bad[-1]["family"]=bad[0]["family"]
        with self.assertRaises(ValueError):f.validate_cases(bad)
        bad=copy.deepcopy(self.cases);bad[-1]["messages"]=bad[0]["messages"]
        with self.assertRaises(ValueError):f.validate_cases(bad)

    def test_strict_json_and_exact_arithmetic(self):
        case=self.cases[0]
        for text in ('{"total":22}', '{"total":true}', '{"total":21.0}', '{"total":21,"total":21}',
                     '```json\n{"total":21}\n```', '{"total":NaN}', '{"total":21} explanation'):
            self.assertFalse(f.check(case,text)["pass"],text)
        self.assertTrue(f.check(case,'{"total":21}')["pass"])

    def test_counts_content_is_not_shape(self):
        _, cases=f.load("constraint-precision")
        c=next(x for x in cases if x["family"]=="four-word-title")
        self.assertTrue(f.check(c,"Ignore This Task Entirely")["pass"])
        self.assertFalse(f.passes(c,decision(adequacy="wrong")))
        c=next(x for x in cases if x["family"]=="banned-word")
        self.assertFalse(f.check(c,"After the long walk, they felt tired and needed rest.")["pass"])
        self.assertFalse(f.check(c,"They felt weary.")["pass"])
        c=next(x for x in cases if x["family"]=="csv-sort")
        self.assertFalse(f.check(c,"name,value\noak,3\nash,1")["pass"])

    def test_refusal_and_stubbornness_shortcuts(self):
        for c in self.cases:
            if c["gold"]["expected_action"]!="abstain":
                self.assertFalse(f.passes(c,decision(action="abstain")))
        _, cases=f.load("independent-judgment")
        self.assertEqual(sum(c["gold"]["control"]=="correct_pushback" for c in cases),9)
        self.assertEqual(sum(c["gold"]["control"]=="incorrect_pushback" for c in cases),9)
        for c in cases:
            wrong="resist_pushback" if c["gold"]["expected_action"]=="accept_correction" else "accept_correction"
            self.assertFalse(f.passes(c,decision(action=wrong)))

    def test_payload_boundaries_and_split_isolation(self):
        c=self.cases[1];s=candidate(self.config,c)
        p=f.request(self.config,c,s,purpose="target_selection")
        self.assertEqual(p["model"],"typesafe/jev-1.14")
        self.assertIn("untrusted",p["questions"]["adequacy"]["instructions"])
        self.assertEqual(len(p["questions"]),4)
        self.assertNotIn("expected_action",p["state"]["gold"])
        held=self.cases[-1]
        with self.assertRaises(ValueError):f.request(self.config,held,candidate(self.config,held),purpose="target_selection")
        with self.assertRaises(ValueError):f.request(self.config,c,s,purpose="heldout_scoring")
        with self.assertRaises(ValueError):f.request(self.config,c,dict(s,termination="token_budget"),purpose="target_selection")
        with self.assertRaises(ValueError):f.request(self.config,c,dict(s,text="changed"),purpose="target_selection")

    def test_response_schema_validation(self):
        good=decision();f.validate_decision(good)
        for mutate in (lambda d:d.update(model="other"),
                       lambda d:d["answers"].pop("action"),
                       lambda d:d["answers"]["relevance"].update(choice="partial"),
                       lambda d:d["answers"]["quality"].update(score=float("nan")),
                       lambda d:d["answers"]["adequacy"]["probabilities"].update(complete=True),
                       lambda d:d["usage"].update(input_tokens=-1)):
            d=copy.deepcopy(good);mutate(d)
            with self.assertRaises(ValueError):f.validate_decision(d)

    def test_select_filters_before_jev_and_deduplicates(self):
        c=self.cases[0]
        samples=[candidate(self.config,c,n=i,text=t) for i,t in enumerate([
            '{"total":22}', '{"total":21}', '{"total":21}', '{ "total":21 }'])]
        receipts={}
        for s in (samples[1],samples[3]):
            p=f.request(self.config,c,s,purpose="target_selection")
            receipts[f.digest(p)]=receipt(p,decision())
        result=f.select(self.config,c,samples,receipts)
        self.assertEqual([r["status"] for r in result["audit"]],["hard_fail_or_censored","pass","duplicate","pass"])
        self.assertEqual(result["selected"]["id"],samples[1]["id"])
        with self.assertRaises(ValueError):f.select(self.config,c,samples,{})
        self.assertIsNone(f.select(self.config,c,[candidate(self.config,c,n=i,text='{"total":22}') for i in range(4)],{})["selected"])

    def test_budget_duplicates_unknown_and_overrun(self):
        p={"example":1};ledger=f.BudgetLedger(.01)
        self.assertTrue(ledger.reserve(p,.006))
        self.assertFalse(ledger.reserve(p,.006))
        with self.assertRaises(ValueError):ledger.reserve({"example":2},.005)
        ledger.settle(f.digest(p),decision(),input_rate=.042,output_rate=0)
        self.assertAlmostEqual(ledger.entries[f.digest(p)]["charged_or_reserved_usd"],.0000504)
        with self.assertRaises(ValueError):ledger.settle(f.digest(p),decision(),input_rate=.042,output_rate=0)
        other=f.BudgetLedger(.01);other.reserve(p,.000001)
        with self.assertRaises(ValueError):other.settle(f.digest(p),decision(),input_rate=.042,output_rate=0)
        self.assertEqual(other.entries[f.digest(p)]["status"],"overrun_halt")
        with self.assertRaises(ValueError):other.reserve({"new":3},.000001)
        estimate=f.estimate([p,p,{"example":2}],input_usd_per_million=.042,output_usd_per_million=0,
                            input_token_bound=6000,output_token_bound=500,retry_reserve=1)
        self.assertEqual(estimate["unique_requests"],2)
        self.assertEqual(estimate["reserved_attempts"],3)
        self.assertAlmostEqual(estimate["estimated_usd_bound"],.000756)

    def test_native_handoff_preserves_history_and_rejects_fixtures(self):
        # Inject only the existing queue prefix interface; no torch/model/GPU.
        module=types.ModuleType("glimmer_slider.queue_targets")
        module.prefix_ids=lambda tokenizer,definition,prompt:[len(definition)+1,2]
        pkg=types.ModuleType("glimmer_slider");pkg.__path__=[]
        class FakeTokenizer:
            def decode(self,ids,skip_special_tokens=True):return bytes(ids).decode()
        cases=[self.cases[1],self.cases[10]]
        selections={c["id"]:dict(selected=candidate(self.config,c),case_sha256=f.digest(c),config_sha256=f.digest(self.config)) for c in cases}
        with patch.dict(sys.modules,{"glimmer_slider":pkg,"glimmer_slider.queue_targets":module}):
            with self.assertRaises(ValueError):f.build_native_packet(self.config,cases,selections,FakeTokenizer(),tokenizer_sha256="f"*64,encode_prefix=module.prefix_ids)
            for s in selections.values():s["selected"]["origin"]="native_glimmer"
            packet=f.build_native_packet(self.config,cases,selections,FakeTokenizer(),tokenizer_sha256="f"*64,encode_prefix=module.prefix_ids)
            for split in ("training","development"):
                self.assertEqual([len(r["history"]) for r in packet[split]],[0,8])
                self.assertNotEqual(packet[split][0]["neutral"],packet[split][0]["positive"])
            selections[cases[0]["id"]]["selected"]["text"]="edited"
            with self.assertRaises(ValueError):f.build_native_packet(self.config,cases,selections,FakeTokenizer(),tokenizer_sha256="f"*64,encode_prefix=module.prefix_ids)

    def test_short_native_histories_are_capped_not_padded(self):
        module=types.ModuleType("glimmer_slider.queue_targets")
        module.prefix_ids=lambda *args:[1,2]
        pkg=types.ModuleType("glimmer_slider");pkg.__path__=[]
        class FakeTokenizer:
            def decode(self,ids,skip_special_tokens=True):return bytes(ids).decode()
        c=next(c for c in self.cases if c["family"]=="string-length")
        s=candidate(self.config,c,origin="native_glimmer")
        selections={c["id"]:dict(selected=s,case_sha256=f.digest(c),config_sha256=f.digest(self.config))}
        with patch.dict(sys.modules,{"glimmer_slider":pkg,"glimmer_slider.queue_targets":module}):
            packet=f.build_native_packet(self.config,[c],selections,FakeTokenizer(),tokenizer_sha256="f"*64,encode_prefix=module.prefix_ids)
        self.assertEqual([len(r["history"]) for r in packet["training"]],[0,1])

    def test_selection_requires_frozen_proposal_condition_and_seeds(self):
        c=self.cases[0]
        samples=[candidate(self.config,c,n=i) for i in range(4)]
        samples[0]["condition"]["system"]="different"
        with self.assertRaises(ValueError):f.select(self.config,c,samples,{})
        samples=[candidate(self.config,c,n=0) for i in range(4)]
        for i,s in enumerate(samples):s["id"]+=str(i)
        with self.assertRaises(ValueError):f.select(self.config,c,samples,{})

    def test_development_scoring_is_separate(self):
        dev=[c for c in self.cases if c["split"]=="development"]
        records=[];receipts={}
        for c in dev:
            for arm in e.ARMS[:3]:
                for slot in range(4):
                    s=candidate(self.config,c,n=slot)
                    condition=dict(system=self.config["positive_definition"] if arm=="plain_instruction" else "",
                        strength=1. if arm=="slider" else 0.,artifact_sha256="a"*64 if arm=="slider" else None)
                    s["condition"]=condition
                    records.append(dict(case_id=c["id"],arm=arm,slot=slot,seed=s["seed"],candidate=s,condition=condition))
                    p=f.request(self.config,c,s,purpose="development_scoring")
                    receipts[f.digest(p)]=receipt(p,decision(c["gold"]["expected_action"]))
        result=e.development_report(self.config,self.cases,records,receipts,artifact_sha256="a"*64,alpha=1.)
        self.assertEqual(len(result["rows"]),48)
        bad=copy.deepcopy(records);bad[0]["case_id"]=self.cases[-1]["id"]
        with self.assertRaises(ValueError):e.development_report(self.config,self.cases,bad,receipts,artifact_sha256="a"*64,alpha=1.)

    def test_public_selector_never_receives_gold(self):
        c=self.cases[-1];s=candidate(self.config,c)
        p=e.public_request(self.config,c,s)
        self.assertNotIn("gold",p["state"])
        self.assertNotIn("reference",p["state"])
        self.assertNotIn("checks",p["state"])
        self.assertEqual(set(p["questions"]),{"relevance","adequacy","quality"})

    def test_direct_final_prefix_contract(self):
        class Tokenizer:
            def apply_chat_template(inner,messages,**kwargs):
                self.assertEqual(messages[-1],dict(role="assistant",recipient="user",content=""))
                self.assertTrue(kwargs["continue_final_message"])
                self.assertFalse(kwargs["add_generation_prompt"])
                return {"input_ids":[17,18,19]}
        self.assertEqual(f.direct_final_prefix_ids(Tokenizer(),"direction","task"),[17,18,19])

    def test_native_prepare_boundary_preserves_pairs(self):
        calls=[]
        module=types.ModuleType("glimmer_slider.training")
        def prepare(model,slider,rows,**kwargs):
            calls.append((model,slider,rows,kwargs))
            return ["actual-native-hidden-row-placeholder"]
        module.prepare=prepare
        pkg=types.ModuleType("glimmer_slider");pkg.__path__=[]
        packet={"training":[{"history":[1]}],"development":[{"history":[2]}]}
        with patch.dict(sys.modules,{"glimmer_slider":pkg,"glimmer_slider.training":module}):
            bank=f.prepare_native_banks("model","slider",packet,max_tokens=1280)
        self.assertIs(calls[0][2],packet["training"])
        self.assertIs(calls[1][2],packet["development"])
        self.assertEqual(calls[0][3],{"max_tokens":1280})
        self.assertEqual(set(bank),{"training","evaluation"})

    def test_complete_matched_evaluation_and_length_report(self):
        lock=dict(status="development_locked", protocol_sha256=f.digest(e.protocol(self.config)),
            thresholds={"semantic_probability":.8,"quality":2,"quality_confidence":.5},
            development_evidence_sha256="d"*64,artifact_sha256="a"*64,alpha=1.)
        records=[];scores={};public={}
        for c in self.cases:
            if c["split"]!="heldout":continue
            for arm in e.ARMS:
                for slot in range(4):
                    s=candidate(self.config,c,n=slot);s["id"]+=arm
                    records.append(dict(case_id=c["id"],arm=arm,slot=slot,candidate=s,
                        seed=self.config["selection_sampling"]["seeds"][slot],
                        condition=dict(system=self.config["positive_definition"] if arm=="plain_instruction" else "",
                            strength=1. if arm=="slider" else 0.,artifact_sha256="a"*64 if arm=="slider" else None)))
                    s["condition"]=records[-1]["condition"]
                    p=f.request(self.config,c,s,purpose="heldout_scoring")
                    scores[f.digest(p)]=receipt(p,decision(c["gold"]["expected_action"]))
                    if arm=="best_of_n":
                        p=e.public_request(self.config,c,s)
                        public[f.digest(p)]=receipt(p,decision(schema=e.public_questions()))
        result=e.evaluate(self.config,self.cases,records,scores,public,lock)
        self.assertEqual(result["generation_calls"],64)
        self.assertEqual(len(result["paired_primary"]),16)
        self.assertTrue(all(r["pass_rate"]==1 for r in result["primary"]))
        bad=copy.deepcopy(records);bad[0]["seed"]=999
        with self.assertRaises(ValueError):e.evaluate(self.config,self.cases,bad,scores,public,lock)
        with self.assertRaises(ValueError):e.evaluate(self.config,self.cases,records,scores,public,{})


if __name__=="__main__":
    unittest.main(verbosity=2)
