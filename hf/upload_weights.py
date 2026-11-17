"""Upload DiffNet weights and config to a public Hugging Face model repo.

Authentication uses the HF_TOKEN environment variable or the credentials
cached by `hf auth login`; the token is never taken as an argument.
HUGGING_FACE_HUB_TOKEN is ignored.
"""
import argparse
import os
import sys
from pathlib import Path

PROTOCOLS = ("doc_protocol", "tdoc_protocol")
PLACEHOLDER_TEXT = {
    "doc_protocol": b"# doc_protocol\n\nWeights trained under the Doc Protocol will be added to this folder.\n",
    "tdoc_protocol": b"# tdoc_protocol\n\nWeights trained under the Syn2Real-TDoc protocol will be added to this folder.\n",
}


def placeholder_path(protocol):
    return f"{protocol}/README.md"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-id", required=True, help="<user>/<name>")
    p.add_argument("--protocol", choices=PROTOCOLS, default="doc_protocol")
    p.add_argument("--weights", type=Path, help="weights-only export")
    p.add_argument("--config", type=Path, help="config matching the weights")
    p.add_argument("--model-card", type=Path, help="uploaded as README.md at the repo root")
    p.add_argument("--init-placeholder", choices=PROTOCOLS, action="append", default=[],
                   help="upload <protocol>/README.md so the folder exists (repeatable)")
    p.add_argument("--dry-run", action="store_true", help="list uploads, no network")
    args = p.parse_args()

    if (args.weights is None) != (args.config is None):
        p.error("--weights and --config must be given together")
    if args.weights is None and args.model_card is None and not args.init_placeholder:
        p.error("nothing to upload")
    if args.weights is not None and args.protocol in args.init_placeholder:
        p.error(f"--init-placeholder {args.protocol} conflicts with uploading {args.protocol} weights")
    if args.repo_id.count("/") != 1:
        p.error("--repo-id must be <user>/<name>")
    for path in (args.weights, args.config, args.model_card):
        if path is not None and not path.is_file():
            p.error(f"file not found: {path}")
    return args


def planned_uploads(args):
    """Return (path_in_repo, source) pairs; source is a Path or bytes."""
    items = []
    if args.weights is not None:
        items.append((f"{args.protocol}/{args.weights.name}", args.weights))
        items.append((f"{args.protocol}/{args.config.name}", args.config))
    if args.model_card is not None:
        items.append(("README.md", args.model_card))
    for protocol in dict.fromkeys(args.init_placeholder):
        items.append((placeholder_path(protocol), PLACEHOLDER_TEXT[protocol]))
    return items


def main():
    args = parse_args()
    items = planned_uploads(args)

    print(f"repo: https://huggingface.co/{args.repo_id} (public model repo)")
    for path_in_repo, src in items:
        if isinstance(src, bytes):
            print(f"  {path_in_repo:<40} <- placeholder ({len(src)} B)")
        else:
            print(f"  {path_in_repo:<40} <- {src} ({src.stat().st_size / 2**20:.1f} MiB)")
    if args.dry_run:
        print("dry run: nothing uploaded")
        return

    os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, get_token

    if get_token() is None:
        sys.exit("no token: export HF_TOKEN=... or run `hf auth login`")

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=False, exist_ok=True)
    if api.repo_info(args.repo_id, repo_type="model").private:
        print(f"warning: {args.repo_id} already exists and is private; make it public on the Hub")

    ops = [
        CommitOperationAdd(path_in_repo=path_in_repo,
                           path_or_fileobj=src if isinstance(src, bytes) else str(src))
        for path_in_repo, src in items
    ]
    if args.weights is not None and api.file_exists(args.repo_id, placeholder_path(args.protocol), repo_type="model"):
        ops.append(CommitOperationDelete(path_in_repo=placeholder_path(args.protocol)))
        print(f"  removing {placeholder_path(args.protocol)} (placeholder)")
    info = api.create_commit(
        repo_id=args.repo_id, repo_type="model", operations=ops,
        commit_message=f"Upload {', '.join(p for p, _ in items)}",
    )
    print(f"done: {info.commit_url}")


if __name__ == "__main__":
    main()
