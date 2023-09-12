name: Bug Report
description: File a bug report
title: "[Bug]: "
labels: ["bug", "triage"]
body:
  - type: markdown
    attributes:
      value: |
        Thanks for taking the time to fill out this bug report!
  - type: input
    id: contact
    attributes:
      label: Email (Optional)
      description: How can we get in touch with you if we need more info?
      placeholder: ex. email@example.com
    validations:
      required: false
  - type: input
    id: version
    attributes:
      label: Version
      description: What version of our software are you running?
      placeholder: v2023.1.1
    validations:
      required: true
  - type: checkboxes
    id: operating-systems
    attributes:
      label: Which OS(es) are you using?
      description: You may select more than one.
      options:
        - label: MacOS
        - label: Windows
        - label: Linux
    validations:
      required: true
  - type: textarea
    id: what-happened
    attributes:
      label: What happened?
      description: Provide a concise description of the bug.
      placeholder: |
        1. I am attempting to do this (code snippet given below).
        2. Expected behavior: ...
        3. Instead, got this behavior: ...
        4. Proposed solution (if any). Even better, you can just submit a PR with your solution to contribute to the community.

        Tip: You can attach images or files by clicking this area and then dragging files in.
    validations:
      required: true
  - type: textarea
    id: snippet
    attributes:
      label: Code snippet
      description: |
        Please provide a code snippet to reproduce the bug. This will be automatically formatted into code, so no need for backticks.
      render: python
  - type: textarea
    id: logs
    attributes:
      label: Log output
      description: |
        Please provide any relevant log output, especially error messages. This will be automatically formatted, so no need for backticks.
      render: shell
  - type: checkboxes
    id: terms
    attributes:
      label: Code of Conduct
      description: By submitting this issue, you agree to follow our [Code of Conduct](https://github.com/materialsproject/.github/blob/main/.github/code_of_conduct.md)
      options:
        - label: I agree to follow this project's Code of Conduct
          required: true
