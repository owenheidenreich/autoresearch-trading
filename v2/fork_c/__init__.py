"""Fork C (Phase 1): supervised classifier for Pickles' directional SPX 0DTE long days.

This package is the behavioral-cloning branch taken after Fork A1 (mechanical
backtest of Row 1) was falsified. It trains a binary classifier on a curated
subset of Pickles' 167 journal days to predict whether he will execute or
explicitly claim to hold a directional SPX 0DTE long call or put on that day,
using only features known by 10:00 ET.

See:
- Plan: ~/.claude/plans/read-this-context-and-snappy-tide.md
- Label rules: v2/fork_c/label_rules.md (lock before parsing)
- Journal source: /Users/gduby/Documents/picklesGPT/pickles/journal/

The label file IS the experiment. Provenance hashes label_rules.md and
tier1_labels.csv alongside the usual dataset/feature/split fingerprints.
"""
