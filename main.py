import os, re, time
from pathlib import Path
from collections import Counter
from difflib import get_close_matches

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from dotenv import load_dotenv
from openai import OpenAI, APIStatusError, InternalServerError
import google.generativeai as genai
import anthropic


load_dotenv()

plt.rcParams.update({
    "font.family": "Times New Roman",
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})


class PilotAudit:
    """
    Collect pilot recommendation outputs from OpenAI, Google Gemini, and Anthropic Claude.
    Saves per-vendor per-category CSVs containing rank1, rank2, rank3 strings.
    """
    SYSTEM_STR = (
        "You are a helpful assistant. Provide ranked recommendations. "
        "Return exactly three model names, best first, one per line. "
        "No extra text, no numbering."
    )

    VENDORS = {
        "openai":    {
            "fn": "_ask_openai",
            "env": "OPENAI_API_KEY",
            "base": "results_openai",
        },
        "google":    {
            "fn": "_ask_gemini",
            "env": "GEMINI_API_KEY",
            "base": "results_google",
        },
        "anthropic": {
            "fn": "_ask_claude",
            "env": "CLAUDE_API_KEY",
            "base": "results_anthropic",
        },
    }

    def __init__(self, run_openai=True, run_gemini=True, run_anthropic=True):
        self.run_flags = {
            "openai": run_openai,
            "google": run_gemini,
            "anthropic": run_anthropic,
        }

    def _extract_top3(self, txt: str):
        out = [ln.strip() for ln in re.split(r"[\r\n]+", txt) if ln.strip()]
        while len(out) < 3:
            out.append(None)
        return tuple(out[:3])

    def _ask_openai(self, prompt, model="gpt-4o", max_retries=3, base_backoff=2.0):
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        for attempt in range(max_retries + 1):
            try:
                rsp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_STR},
                        {"role": "user",   "content": f"{prompt} Top 3 only."}
                    ]
                )
                return rsp.choices[0].message.content.strip()
            except (InternalServerError, APIStatusError) as err:
                if not (500 <= err.status_code < 600):
                    raise
                if attempt == max_retries:
                    raise
                wait = base_backoff * (2 ** attempt)
                print(f"OpenAI 5xx (attempt {attempt+1}/{max_retries}) retrying in {wait:.0f}s")
                time.sleep(wait)
        raise RuntimeError("OpenAI retry loop failed")

    def _ask_gemini(self, prompt):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        mdl = genai.GenerativeModel("gemini-2.0-flash")
        full = f"{self.SYSTEM_STR}\n\n{prompt}"
        return (mdl.generate_content(full).text or "").strip()

    def _ask_claude(self, prompt):
        client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
        msg = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=200,
            system=self.SYSTEM_STR,
            messages=[{"role": "user", "content": prompt}]
        )
        return "".join(getattr(b, "text", str(b)) for b in msg.content).strip()

    def run_prompt_iterations(self, prompt, category, vendor, n_iter, addition):
        cfg = self.VENDORS[vendor]
        key = os.getenv(cfg["env"])
        if not key:
            print(f"{cfg['env']} not set skipping {vendor}")
            return

        rows = []
        ask_fn = getattr(self, cfg["fn"])
        for _ in range(n_iter):
            raw = ask_fn(prompt)
            rank1, rank2, rank3 = self._extract_top3(raw)
            rows.append({"rank1": rank1, "rank2": rank2, "rank3": rank3})
            time.sleep(0.4)

        out_dir = Path(f"{cfg['base']}_{category.lower().replace(' ', '_')}")
        out_dir.mkdir(parents=True, exist_ok=True)

        slug = re.sub(r"[^a-z0-9]+", "_", prompt.lower())[:30] or "prompt"
        out_file = out_dir / f"{n_iter}_{slug}_{addition}.csv"
        pd.DataFrame(rows).to_csv(out_file, index=False)
        print(f"WORKING {vendor:9s} -> {out_file}")

    def run_all(self, prompt: str, category: str, iter_range: range = range(5, 51, 3)):
        vendors = [v for v, flag in self.run_flags.items() if flag]
        if not vendors:
            print("Nothing to run all vendor flags False")
            return
        for n in iter_range:
            for v in vendors:
                self.run_prompt_iterations(prompt, category, v, n, category)


class PilotAnalysis:
    """
    Read pilot outputs and produce variation plots for rank1, rank2, rank3
    across increasing iteration counts.
    """
    ALIAS_MAP = {
        "chatgpt4": "GPT-4o (Nov '24)",
        "gpt4": "GPT-4o (Nov '24)",
        "gpt40613": "GPT-4o (Nov '24)",
        "gpt41106preview": "GPT-4o (Nov '24)",
        "gpt4openai": "GPT-4o (Nov '24)",
        "gpt4fromopenai": "GPT-4o (Nov '24)",
        "gpt35": "GPT-4o mini",
        "gpt40": "GPT-4o (Nov '24)",
        "gpt432k": "GPT-4o (Nov '24)",
        "gpt4api": "GPT-4o (Nov '24)",
        "gpt4omini": "GPT-4o mini",
        "gpt4o": "GPT-4o (Nov '24)",
        "gpt4turbo": "GPT-4o (Nov '24)",
        "gpt4turbopreview": "GPT-4o (Nov '24)",
        "chatgpt4o": "GPT-4o (Nov '24)",
        "chatgpt35turbo": "GPT-4o mini",
        "gpt35turbo": "GPT-4o mini",
        "claude3": "Claude 3 Opus",
        "claudeopus": "Claude 3 Opus",
        "claude3opus": "Claude 3 Opus",
        "claude": "Claude 3.7 Sonnet",
        "claude37": "Claude 3.7 Sonnet",
        "claude37sonnet": "Claude 3.7 Sonnet",
        "sonnet": "Claude 3.7 Sonnet",
        "haiku": "Claude 3.5 Haiku",
        "claude3haiku": "Claude 3.5 Haiku",
        "gemini": "Gemini 2.5 Pro",
        "geminipro": "Gemini 2.5 Pro",
        "gemini15pro": "Gemini 1.5 Pro (Sep)",
        "geminiflash": "Gemini 2.0 Flash",
        "bard": "Gemini 1.0 Pro",
    }

    _norm_re = re.compile(r"[^a-z0-9]+")

    def __init__(self, folder_path, output_name, title):
        self.folder = folder_path
        self.output = output_name
        self.title = title
        os.makedirs(self.folder, exist_ok=True)
        self.visual_dir = "pilot_output"
        os.makedirs(self.visual_dir, exist_ok=True)

    def norm(self, txt):
        if not txt:
            return ""
        cleaned = self._norm_re.sub("", str(txt).lower())
        return cleaned.split("by")[0].strip()

    def to_canonical(self, raw):
        key = self.norm(raw)
        return self.ALIAS_MAP.get(key, raw)

    def column_variation(self, series: pd.Series):
        clean = series.dropna().map(self.to_canonical)
        if clean.empty:
            return 0.0
        freq = Counter(clean)
        return 1.0 - (max(freq.values()) / len(clean))

    def make_plot(self):
        files = [f for f in os.listdir(self.folder) if re.match(r"\d+_", f)]
        results = []

        leaderboard = pd.read_csv("ai_leaderboard_final_clean.csv")
        model_to_company = dict(zip(leaderboard["Model"], leaderboard["Creator"]))

        for fname in files:
            m = re.match(r"(\d+)_", fname)
            if not m:
                continue
            iters = int(m.group(1))

            try:
                df = pd.read_csv(os.path.join(self.folder, fname))
            except Exception:
                continue

            cols = {c.strip().lower(): c for c in df.columns}
            if not all(k in cols for k in ["rank1", "rank2", "rank3"]):
                continue

            r1 = df[cols["rank1"]].map(self.to_canonical).map(model_to_company).fillna("Unknown")
            r2 = df[cols["rank2"]].map(self.to_canonical).map(model_to_company).fillna("Unknown")
            r3 = df[cols["rank3"]].map(self.to_canonical).map(model_to_company).fillna("Unknown")

            v1 = self.column_variation(r1)
            v2 = self.column_variation(r2)
            v3 = self.column_variation(r3)
            results.append((iters, v1, v2, v3))

        if not results:
            print(f"No CSV files in '{self.folder}'")
            return

        results.sort(key=lambda x: x[0])
        x, y1, y2, y3 = zip(*results)

        plt.figure(figsize=(10, 6))
        plt.plot(x, y1, marker="o", label="Rank 1")
        plt.plot(x, y2, marker="s", label="Rank 2")
        plt.plot(x, y3, marker="^", label="Rank 3")
        plt.xlabel("Number of iterations")
        plt.ylabel("Variation")
        plt.title(self.title)
        plt.ylim(0, 1.05)
        plt.xticks(x)
        plt.grid(True, alpha=0.4)
        plt.legend()

        out_path = os.path.join(self.visual_dir, f"{self.output}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=120)
        plt.close()
        print(f"variation plot saved to {out_path}")


def main():
    runner = PilotAudit(
        run_openai=True,
        run_gemini=True,
        run_anthropic=True
    )

    pilot_jobs = [
        ("GeneralQuestions", "Which AI is best for drafting well structured essays?"),
        ("Math", "I'm building a tutor for high school math. Which LLM should I use?")
    ]

    for category, prompt in pilot_jobs:
        runner.run_all(prompt, category)

    categories = [
        ("generalquestions", "GeneralQuestions"),
        ("math", "Math"),
    ]
    vendors = [
        ("google", "Google"),
        ("openai", "OpenAI"),
        ("anthropic", "Anthropic"),
    ]

    for vendor_key, vendor_name in vendors:
        for cat_key, cat_name in categories:
            folder = f"results_{vendor_key}_{cat_key}"
            output_name = f"{vendor_key}_{cat_key}"
            title = f"{vendor_name} {cat_name}"
            pilot = PilotAnalysis(folder, output_name, title)
            pilot.make_plot()


if __name__ == "__main__":
    main()
