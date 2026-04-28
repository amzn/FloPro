# Flo Pro

> **[WIP]** This repository is a work in progress. Documentation is still evolving.

Flo Pro is a distributed negotiation framework where vendors and Amazon
iteratively agree on purchase order plans through an ADMM-style coordination
loop. This monorepo hosts the Flo Pro SDK, the Flo Pro ADK, and the shared
documentation site.

## Components

- **[`python/flo-pro-sdk`](python/flo-pro-sdk)** — Flo Pro SDK. Defines the
  agent contract (`AgentDefinition`, `Solution`, `Objective`) and the message
  types exchanged during negotiation. Both vendor agents and the Amazon agent
  build on this.
- **[`python/flo-pro-adk`](python/flo-pro-adk)** — Flo Pro ADK (Vendor Agent
  Development Kit). Ships a simulated Amazon counterparty, a solver framework,
  test runners, and ready-to-run scenarios so vendors can build and validate
  their agent locally. See the package [README](python/flo-pro-adk/README.md)
  and [API reference](python/flo-pro-adk/docs/api/README.md).
- **[`docs/`](docs)** — MkDocs site combining Flo Pro SDK and Flo Pro ADK
  documentation.
- **[`examples/`](examples)** — Sample vendor agents and end-to-end scripts.
- **[`tools/`](tools)** — Developer tooling and helper scripts.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
