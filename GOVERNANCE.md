# Governance

Undercurrent is an open-source project sponsored by Wrynx.
This document describes how decisions are made and how contributors can take
on more responsibility. The model is intentionally lightweight: Wrynx
maintainers hold final say while the project is young, and there is a written
path for anyone outside Wrynx to become a maintainer.

There is no foundation, steering committee or technical oversight board behind
Undercurrent. If that changes, this document will change with it.

## Roles

**Users** install and use Undercurrent. They help the project by reporting
bugs, asking and answering questions in
[Discussions](https://github.com/wrynx/undercurrent/discussions), and telling
us what they need.

**Contributors** are anyone who has had a contribution merged: code, docs,
tests, examples, issue triage or reviews. Anyone can become a contributor by
following [CONTRIBUTING.md](CONTRIBUTING.md).

**Maintainers** are listed in [MAINTAINERS.md](MAINTAINERS.md). They:

- review and merge pull requests in their areas,
- triage issues and keep the [roadmap](ROADMAP.md) current,
- cut releases (see [RELEASING.md](RELEASING.md)),
- uphold the [Code of Conduct](CODE_OF_CONDUCT.md).

**The lead maintainer** is a maintainer who also acts as tie-breaker (see
below) and is responsible for the project's overall direction. While the
project is young, the lead maintainer is a Wrynx employee, appointed by Wrynx.

## Decision-making

Most decisions happen in pull requests and issues, by **lazy consensus**:

1. A change is proposed in a pull request (or, for larger ideas, an issue or
   Discussion first).
2. It needs approval from at least one maintainer who did not author it.
3. If no maintainer objects within a reasonable time for its size (a few
   working days for anything non-trivial), it can be merged.

An objection should explain what would make the change acceptable. Most
disagreements are resolved by discussion.

**Contentious changes.** If maintainers can't reach consensus, any maintainer
can call a vote on the pull request or issue. Each maintainer has one vote; a
simple majority of the maintainers who vote decides. Votes stay open for at
least one week so everyone can take part.

**Tie-breaker.** If a vote is tied, or a decision is blocking and can't wait,
the lead maintainer decides and records the reasoning in the thread.

Changes that usually warrant an issue or Discussion before a pull request:
breaking changes to the public API or the YAML spec format, new engine
adapters, new runtime dependencies, and changes to this document.

## Becoming a maintainer

Maintainers are chosen for sustained, high-quality contributions, not for
employer or volume of commits. Signals we look for:

- a track record of merged contributions over several months,
- thoughtful code reviews and help for other users and contributors,
- good judgement about the project's scope, stability and users,
- conduct consistent with the [Code of Conduct](CODE_OF_CONDUCT.md).

The process:

1. An existing maintainer nominates the candidate (self-nomination by asking a
   maintainer is fine) privately to the other maintainers, with links to their
   contributions.
2. Existing maintainers discuss the nomination. It is approved by lazy
   consensus, or by a vote under the rules above if anyone objects.
3. If approved and the candidate accepts, a pull request adds them to
   [MAINTAINERS.md](MAINTAINERS.md) with their area of focus.

New maintainers may start with responsibility for one area (for example, an
engine adapter or the sinks) and grow from there.

## Inactivity and stepping down

Maintainers can step down at any time by opening a pull request against
[MAINTAINERS.md](MAINTAINERS.md).

A maintainer with no project activity (reviews, commits, issue or Discussion
participation) for **12 months** moves to the emeritus list. Emeritus
maintainers are thanked for their work and can return by asking any active
maintainer; returning follows the nomination process, usually as a formality.

## Code of Conduct

Everyone taking part in the project follows the
[Code of Conduct](CODE_OF_CONDUCT.md). Report conduct issues to
conduct@wrynx.com. Report security issues as described in
[SECURITY.md](SECURITY.md), not in public issues.

## Changing this document

Changes to this governance document are made by pull request. They are
announced in [Discussions](https://github.com/wrynx/undercurrent/discussions),
stay open for at least two weeks, and need approval from a majority of the
maintainers.
