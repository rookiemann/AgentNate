# System Prompt Manager - Implementation Plan

## Overview
Add system prompt management to the chat tab with:
- Clear chat button
- System prompt selector dropdown
- Save/edit custom prompts
- AI-assisted prompt generation
- Pre-built library of prompts

---

## 1. UI Changes (index.html)

### Chat Header Bar (new)
Add a toolbar above the chat messages area:
```html
<div class="chat-header">
    <div class="system-prompt-selector">
        <button id="system-prompt-btn" class="btn-secondary">
            <span id="current-prompt-name">Default Assistant</span>
            <span class="dropdown-arrow">▼</span>
        </button>
        <div id="system-prompt-dropdown" class="dropdown-menu hidden">
            <!-- Populated by JS -->
        </div>
    </div>
    <div class="chat-actions">
        <button id="clear-chat-btn" class="btn-icon" title="Clear chat">🗑️</button>
        <button id="edit-prompt-btn" class="btn-icon" title="Edit system prompt">✏️</button>
    </div>
</div>
```

### System Prompt Modal (new)
Modal for viewing/editing/creating prompts:
```html
<div id="prompt-modal" class="modal hidden">
    <div class="modal-content prompt-modal-content">
        <div class="modal-header">
            <h2 id="prompt-modal-title">Edit System Prompt</h2>
            <button class="btn-close" onclick="closePromptModal()">×</button>
        </div>
        <div class="modal-body">
            <!-- Tabs: Library | Custom | Generate -->
            <div class="prompt-tabs">
                <button class="prompt-tab active" data-tab="library">Library</button>
                <button class="prompt-tab" data-tab="custom">My Prompts</button>
                <button class="prompt-tab" data-tab="generate">AI Generate</button>
            </div>

            <!-- Library Tab -->
            <div id="prompt-library-tab" class="prompt-tab-content active">
                <div class="prompt-categories">
                    <!-- Category buttons -->
                </div>
                <div class="prompt-list" id="library-prompt-list">
                    <!-- Prompt cards -->
                </div>
            </div>

            <!-- Custom Tab -->
            <div id="prompt-custom-tab" class="prompt-tab-content hidden">
                <div class="prompt-list" id="custom-prompt-list">
                    <!-- User's saved prompts -->
                </div>
                <button id="new-prompt-btn" class="btn-primary">+ New Prompt</button>
            </div>

            <!-- Generate Tab -->
            <div id="prompt-generate-tab" class="prompt-tab-content hidden">
                <div class="generate-prompt-form">
                    <label>Describe what you want the assistant to do:</label>
                    <textarea id="prompt-description" placeholder="e.g., A Python expert that explains concepts simply and provides working code examples..."></textarea>
                    <button id="generate-prompt-btn" class="btn-primary">Generate with AI</button>
                </div>
                <div id="generated-prompt-result" class="hidden">
                    <label>Generated Prompt (edit if needed):</label>
                    <textarea id="generated-prompt-text"></textarea>
                    <div class="prompt-save-row">
                        <input type="text" id="generated-prompt-name" placeholder="Prompt name...">
                        <button id="save-generated-btn" class="btn-primary">Save</button>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
```

### Prompt Editor Modal (for editing single prompt)
```html
<div id="prompt-editor-modal" class="modal hidden">
    <div class="modal-content">
        <div class="modal-header">
            <h2>Edit Prompt</h2>
            <button class="btn-close" onclick="closePromptEditor()">×</button>
        </div>
        <div class="modal-body">
            <div class="form-group">
                <label>Name</label>
                <input type="text" id="edit-prompt-name">
            </div>
            <div class="form-group">
                <label>Category</label>
                <select id="edit-prompt-category">
                    <option value="general">General</option>
                    <option value="coding">Coding</option>
                    <option value="writing">Writing</option>
                    <option value="business">Business</option>
                    <option value="education">Education</option>
                    <option value="creative">Creative</option>
                    <option value="custom">Custom</option>
                </select>
            </div>
            <div class="form-group">
                <label>System Prompt</label>
                <textarea id="edit-prompt-content" rows="10"></textarea>
            </div>
        </div>
        <div class="modal-footer">
            <button class="btn-secondary" onclick="closePromptEditor()">Cancel</button>
            <button class="btn-danger" id="delete-prompt-btn">Delete</button>
            <button class="btn-primary" id="save-prompt-btn">Save</button>
        </div>
    </div>
</div>
```

---

## 2. JavaScript (app.js)

### State
```javascript
// System prompt state
let systemPrompts = [];           // All prompts (library + custom)
let customPrompts = [];           // User's saved prompts (from localStorage)
let activePromptId = 'default';   // Currently selected prompt ID
let activePromptContent = '';     // The actual prompt text being used
```

### LocalStorage Keys
- `agentNate_customPrompts` - Array of user's custom prompts
- `agentNate_activePromptId` - ID of currently active prompt

### Functions to Add
```javascript
// Initialization
function initSystemPrompts()
function loadCustomPrompts()
function saveCustomPrompts()

// UI
function openPromptModal()
function closePromptModal()
function renderPromptLibrary(category)
function renderCustomPrompts()
function selectPrompt(promptId)
function updatePromptDisplay()

// CRUD
function createNewPrompt()
function editPrompt(promptId)
function savePrompt(promptData)
function deletePrompt(promptId)

// AI Generation
function generatePromptWithAI()
function saveGeneratedPrompt()

// Chat
function clearChat()
function getSystemPromptForChat()  // Returns active prompt content
```

### Modify sendMessage()
Update to prepend system prompt to messages:
```javascript
async function sendMessage() {
    // ... existing code ...

    // Build messages array with system prompt first
    const messagesForAPI = [];

    // Add system prompt if set
    const systemPrompt = getSystemPromptForChat();
    if (systemPrompt) {
        messagesForAPI.push({
            role: 'system',
            content: systemPrompt
        });
    }

    // Add conversation history
    messagesForAPI.push(...messages);

    // Send via WebSocket
    // ...
}
```

---

## 3. CSS (styles.css)

### New Styles Needed
```css
/* Chat Header */
.chat-header { ... }
.system-prompt-selector { ... }
.dropdown-menu { ... }

/* Prompt Modal */
.prompt-modal-content { ... }
.prompt-tabs { ... }
.prompt-tab { ... }
.prompt-tab-content { ... }
.prompt-categories { ... }
.prompt-list { ... }
.prompt-card { ... }

/* Prompt Editor */
.prompt-save-row { ... }

/* Generate Form */
.generate-prompt-form { ... }
```

---

## 4. System Prompt Library

### Categories & Prompts

#### General (5 prompts)
1. **Default Assistant** - Helpful, balanced general assistant
2. **Concise** - Brief, to-the-point responses
3. **Detailed Explainer** - Thorough explanations with examples
4. **Socratic Teacher** - Answers with guiding questions
5. **Devil's Advocate** - Challenges assumptions constructively

#### Coding (8 prompts)
1. **Code Assistant** - General programming help
2. **Python Expert** - Python-focused with best practices
3. **JavaScript/TypeScript** - Modern JS/TS development
4. **Code Reviewer** - Reviews code for issues and improvements
5. **Debugger** - Focuses on finding and fixing bugs
6. **Algorithm Designer** - Data structures and algorithms
7. **System Architect** - High-level design and architecture
8. **DevOps Engineer** - CI/CD, Docker, infrastructure

#### Writing (6 prompts)
1. **Writing Coach** - Improves writing style and clarity
2. **Copywriter** - Marketing and persuasive copy
3. **Technical Writer** - Documentation and technical content
4. **Editor** - Proofreading and refinement
5. **Storyteller** - Creative fiction and narratives
6. **Academic Writer** - Research papers and formal writing

#### Business (5 prompts)
1. **Business Analyst** - Strategy and analysis
2. **Product Manager** - Product thinking and roadmaps
3. **Marketing Strategist** - Marketing campaigns and strategy
4. **Sales Coach** - Sales techniques and pitches
5. **Startup Advisor** - Entrepreneurship guidance

#### Creative (5 prompts)
1. **Creative Director** - Ideation and creative concepts
2. **Worldbuilder** - Fictional worlds and settings
3. **Character Designer** - Character development
4. **Brainstorm Partner** - Rapid ideation
5. **Game Designer** - Game mechanics and design

#### Education (4 prompts)
1. **Tutor** - Patient teaching for any subject
2. **Language Teacher** - Language learning focus
3. **Math Mentor** - Mathematics explanations
4. **Science Explainer** - Scientific concepts

#### Productivity (4 prompts)
1. **Task Planner** - Breaking down projects
2. **Meeting Facilitator** - Agendas and summaries
3. **Decision Helper** - Pros/cons analysis
4. **Goal Coach** - Goal setting and tracking

#### Specialized (5 prompts)
1. **Legal Assistant** - Legal document review (with disclaimer)
2. **Data Analyst** - Data analysis and visualization
3. **UX Designer** - User experience design
4. **Research Assistant** - Research and synthesis
5. **Translator** - Translation between languages

**Total: ~42 pre-built prompts**

---

## 5. Meta-Prompt for AI Generation

```
You are a system prompt engineer. Your task is to write effective system prompts for AI assistants.

Given a description of what the user wants, create a well-structured system prompt that:
1. Clearly defines the AI's role and expertise
2. Sets the appropriate tone and communication style
3. Includes specific behaviors and guidelines
4. Mentions any constraints or boundaries
5. Is concise but comprehensive (typically 100-300 words)

User's description: {user_input}

Write only the system prompt, nothing else. Do not include quotes around it or any preamble.
```

---

## 6. Files to Modify

| File | Changes |
|------|---------|
| `ui/index.html` | Add chat header, prompt modal, prompt editor modal |
| `ui/app.js` | Add prompt management state and functions, modify sendMessage |
| `ui/styles.css` | Add styles for new UI components |
| `ui/prompts.js` | NEW FILE - Contains prompt library data |

---

## 7. Implementation Order

1. **Phase 1: Clear Chat** (simple win)
   - Add clear button to UI
   - Implement clearChat() function

2. **Phase 2: Prompt Library Data**
   - Create prompts.js with all pre-built prompts
   - Define data structure

3. **Phase 3: Basic Prompt Selection**
   - Add chat header with dropdown
   - Implement prompt selection
   - Modify sendMessage to include system prompt
   - LocalStorage for active prompt

4. **Phase 4: Prompt Modal UI**
   - Create modal HTML structure
   - Add CSS styling
   - Render library prompts by category

5. **Phase 5: Custom Prompts**
   - LocalStorage CRUD for custom prompts
   - Edit/delete functionality
   - Custom prompts tab

6. **Phase 6: AI Generation**
   - Generate tab UI
   - Integration with loaded model
   - Save generated prompts

---

## 8. Data Structure

### Prompt Object
```javascript
{
    id: 'python-expert',           // Unique identifier
    name: 'Python Expert',         // Display name
    category: 'coding',            // Category for filtering
    description: 'Python development with best practices', // Short description
    content: '...',                // The actual system prompt text
    isBuiltIn: true,               // true for library, false for custom
    icon: '🐍'                     // Optional emoji icon
}
```

### LocalStorage Structure
```javascript
// agentNate_customPrompts
[
    { id: 'my-prompt-1', name: '...', category: 'custom', content: '...', isBuiltIn: false }
]

// agentNate_activePromptId
'python-expert'
```

---

## 9. Verification

1. Clear chat button works and resets conversation
2. Can select prompts from library dropdown
3. Selected prompt persists on page reload
4. Can create, edit, delete custom prompts
5. Custom prompts persist in localStorage
6. AI generation works with loaded model
7. System prompt is correctly prepended to API calls
8. All 40+ library prompts are available and categorized
