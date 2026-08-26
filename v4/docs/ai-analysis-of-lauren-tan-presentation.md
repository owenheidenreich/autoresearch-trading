AI Notes: Building Trust and Efficiency with Code Generation Agents


This presentation explores the critical aspect of building trust in AI agents for code generation. Lauren Tan, with her extensive experience at Meta and Netflix, shares insights on how to move from heavily supervised agent interaction to a state where agents can autonomously perform tasks like merging pull requests. The core theme revolves around developing verification skills for agents and understanding how to effectively manage and guide them, drawing parallels to engineering management principles.








Introduction to Agent Trust


Lauren Tan introduces the challenge of trusting AI agents in coding, especially for experienced engineers who have developed strong opinions on good engineering practices. She highlights the common issue of agents "hallucinating" or confidently presenting incorrect information, which erodes trust and limits their utility. The core problem is that agents often provide incorrect information, leading to a lack of trust. This is analogous to how an untrusting manager resorts to micromanagement, hindering team productivity .








The Trust Curve and Productivity


Lauren illustrates her journey with agents using a "trust curve." Initially, she was heavily involved in every step of the agent's process due to a lack of trust. As her confidence in the agents grew, she was able to delegate more complex tasks. In the early stages, there was a high level of human involvement, requiring constant prompting and verification. Currently, agents can auto-merge PRs, demonstrating a high level of trust. This increased trust has directly correlated with a significant boost in her productivity, as evidenced by the number of PRs landed at Cursor. Her productivity saw an exponential increase in recent months, reaching nearly 800 PRs in the first 12 days of the current month, after an initial lower productivity period due to codebase unfamiliarity .


 SCREENSHOT 








The Power of Verification Skills


A key skill for effective agent interaction is verification. This involves enabling agents to run code, take CPU traces, heap snapshots, or interact with simulators to test and confirm their output. Verification closes the loop; it doesn't guarantee good code, but ensures correctness. An example is the "Agent Window" at Cursor, a React application. Lauren recounts an early experience where debugging this window was a slow, manual process, with her acting as the bottleneck by manually interpreting Chrome DevTools traces and relaying information to the agent .


 SCREENSHOT 








The Feature Map: Guiding Agents


To overcome the limitations of agents lacking context, the concept of a "feature map" was introduced. This map teaches the agent how to navigate and understand different features within an application. The problem was that agents struggled to locate specific UI elements or features based on vague descriptions. The solution is a feature map that provides context, enabling agents to understand user reports and screenshots more effectively. The P-SAC plugin includes a "create verification skill" that helps generate these feature maps, detailing how to access features, keyboard shortcuts, and even DOM elements for programmatic control .


 SCREENSHOT 








Understanding P-SAC and Skill Development


P-SAC (Potato Sack) is a plugin developed by Lauren, inspired by Gary Tan's "G Stack." It's designed to incorporate her practical engineering practices and skills for agents. It started incrementally by observing agent failure modes. The "Howl" skill addresses agents confidently stating incorrect information without actually reading the relevant code. The plugin aims to improve agent intelligence by providing high-quality instructions and context, similar to how a manager guides a new engineer .


 SCREENSHOT 








Maintaining Skills and Ensuring Verification Quality


Maintaining these agent skills is crucial as the codebase evolves. Lauren plans to discuss how these skills are kept up-to-date, likely through a system of "evals" which function as unit tests for agents. The challenge is keeping agent skills relevant with product changes. The concept of "evals" is introduced as a method for testing agent performance, analogous to unit tests for traditional code .


 SCREENSHOT 








Maintaining Agent Skills and Verification


Lauren Tan explains that "evals" serve as a unit test for AI agents, allowing developers to rigorously test and maintain their skills. These evals can be custom-built, with a more rigorous approach detailed in the "Eval Playbook" within P Stack. The process involves spawning sub-agents to evaluate the main agent's performance against a predefined rubric . To prevent agents from altering their behavior when aware of evaluation, individual directories are created with names that obscure the evaluation purpose . Cursor's support for multiple models allows for evaluating skills across different platforms, providing insights into performance variations . Maintaining these skills requires a "backseat driver" mentality, observing and questioning the agent's actions to identify areas for improvement .


 SCREENSHOT 








The Role of Observation and Iteration


Lauren emphasizes the importance of observing agent behavior, especially when building custom skills. This involves closely monitoring tool calls and thinking blocks to pinpoint where agents falter. Starting locally is recommended for building verification skills, allowing direct observation of agent-application interactions . Cloud agents, particularly within Cursor, offer significant advantages for scaling these verification and control skills, benefiting the entire team and company . An example is provided of an agent named "Benny" that automatically reproduces bugs from reports, confirming if they are already fixed . This saves significant time and provides immediate feedback.


 SCREENSHOT 








Trust and the Iterative Journey


Building trust in AI-driven verification is presented as an iterative journey. The process involves starting with local verification and gradually scaling to cloud-based solutions. The journey from using a few agents to a highly automated system requires significant investment and observation, with no shortcuts . Plugins like P Stack can accelerate this process, but ultimately, trust is built through personal investment in developing and understanding these skills . Encoding individual engineering standards and preferences into agent skills allows for a more personalized and trustworthy automation process .


 SCREENSHOT 








Refactoring and Rewriting in the Age of AI


The discussion shifts to the controversial topic of refactoring and rewriting codebases, arguing that AI agents can make these processes more viable and beneficial. Traditionally, engineers are discouraged from large-scale rewrites due to perceived risks and costs . However, Lauren argues that "brownfield" applications, especially those with existing infrastructure and guardrails, are well-suited for AI-driven refactoring . Big tech companies often build robust infra to accommodate varying skill levels, which inadvertently provides guardrails for AI agents . "Greenfield" applications, or entirely new projects, present both the greatest risk and opportunity. "Vibe coded" applications without strong constraints can become unmanageable .


 SCREENSHOT 








Organic Architecture and Agent Empowerment


The concept of "organic architecture" is introduced, describing codebases that evolve organically without strong initial constraints, often leading to complexity that even agents struggle to manage. Starting a codebase with strong constraints is crucial for enabling agents to write good code and for building trust in the system . Lauren shares her experience refactoring Grokbot, resulting in over six hundred PRs merged by agents, demonstrating a high level of automation and trust . This level of automation empowers not only engineers but also designers, product managers, and GTM teams to contribute features without the fear of introducing regressions . The CI process for Grokbot has strong constraints, making manual coding challenging but absorbing this complexity for agents .


 SCREENSHOT 








PR Size and CI/CD Practices


Colin inquires about the typical size of Pull Requests (PRs) and the nature of the Continuous Integration (CI) process. Lauren explains that PR sizes can vary significantly, from fifty to a thousand lines, depending on the task. She encourages splitting work into multiple PRs to maintain a rich and atomic history, which aids in debugging and reverting changes . The CI process for Grokbot, codenamed "Dune," is described as "annoying" due to its strict checks .


 SCREENSHOT 








Strict CI Checks and Banned Practices


The CI for Grokbot implements several strict rules to prevent common pitfalls in agent-generated code. A known "footgun" in React,  useEffect , is explicitly banned in Dune and Grokbot . Code comments are also banned because agents often generate irrelevant or misleading historical context within them . The CI aims to prevent performance issues, which are a common problem in agent windows due to poor process isolation .


 SCREENSHOT 








Architectural Enforcement and Framework Design


Grokbot's architecture is designed to enforce best practices through strict rules and conventions, making it easier for agents to write quality code. The framework is built on the principle that "the shortest path is the best path," guiding agents to follow the most efficient and correct way to solve problems . Features are organized into single directories, with all related code co-located, simplifying navigation and development for agents . The CI enforces rules around the dependency graph, preventing accidental imports between different directories (e.g., Electron main vs. Electron renderer) .


 SCREENSHOT 








Layers of Enforcement


Lauren emphasizes a layered approach to code quality enforcement, moving beyond soft checks to hard, automated rules. CI checks, lints, and compiler diagnostics provide hard constraints that prevent agents from writing substandard code . Rules in tools like Bugbot and style guides are considered "soft" and should not be the sole means of enforcement, as they can be easily bypassed . The goal is to transform human code review comments into automated rules or CI failures, thereby eliminating the need for manual oversight .


 SCREENSHOT 








Token Usage and ROI


Addressing concerns about token usage, Lauren acknowledges that while her team has unlimited tokens, the principles can be applied cost-effectively. Investing in setting up a robust codebase with strict enforcement requires an upfront token investment but pays dividends in the long run . This approach empowers not only developers but also PMs and designers to contribute sustainably, even with less experienced agents . The release of Grok 4.6 offers enhanced intelligence at the same token cost as previous versions, optimizing the cost-intelligence frontier .


 SCREENSHOT 








Product Team Adaptation and Grokbot's Accessibility


The rapid pace of development enabled by agents necessitates adaptation from other company functions. Grokbot serves as a cultural shift, making agent technology accessible and user-friendly for non-technical roles, resembling familiar interfaces like iMessage . Prior to Grokbot, tools like Cursor were primarily developer-centric, making them less delightful for other departments .


 SCREENSHOT 








Grokbot's Impact on Accessibility and User Experience


Grokbot has significantly improved the accessibility of AI tools, especially for non-technical users. Previously, tools like Cursor were primarily designed for developers, offering powerful but complex interfaces. Grokbot, however, provides a more intuitive and familiar user experience, resembling messaging applications like iMessage. Grokbot is described as a "culture moment for people who aren't in tech" . Its interface is "very, very accessible" and "comfortable, very familiar" . This approach allows individuals in roles like GTM (Go-To-Market) and product management to engage with AI agents effectively, transforming their workflow.








Streamlining Workflows and Collaboration


The introduction of Grokbot has enabled new forms of orchestration and collaboration. Users can assign specific agents to manage tasks or accounts, and product managers can leverage agents to summarize work, providing clear visibility into team activities. Agents can be assigned per account or used to summarize individual contributions, such as "an agent that summarizes all the work that Lauren did last night" . This capability allows PMs to stay informed about team progress and contributions.








Empowering Non-Expert Contributors and Accelerating Development


A key benefit highlighted is Grokbot's ability to empower individuals who are not expert engineers to contribute to the development process. The "Dune architecture" and its strict constraints enable this, allowing designers and PMs to directly ship features . This significantly speeds up the development cycle for teams like the Grokbot team, enabling them to "ship so quickly" . Even bug fixes submitted by non-experts are often "perfect," requiring minimal review .








Future Plans and Call to Action


The team has ambitious plans for future developments and encourages users to engage with Grokbot and provide feedback. There is "a lot planned" for future releases . The speaker invites users to try Grokbot, explore Cursor, and offer feedback . For further questions, users are encouraged to reach out via direct message on Twitter or potentially through future Twitter Spaces .








Conclusion and Key Takeaways


Grokbot represents a significant advancement in making AI tools accessible and practical for a wider audience. By providing a familiar interface and enabling non-expert contributions, it streamlines workflows, enhances collaboration, and accelerates development cycles. The "Dune architecture" plays a crucial role in ensuring the quality and effectiveness of these contributions. The team is actively developing new features and welcomes user feedback to further improve the platform.

