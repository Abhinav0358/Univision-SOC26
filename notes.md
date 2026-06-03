# SOC Project - Week 1 Technical Notes

This document summarizes the core logic, testing patterns, and architecture structures required for the development of our automated computer vision pipeline.

---

## 1. Camera Logic & Image Filtering

### 1.1 Motion-Activated Camera Flowchart

To minimize storage and computing costs, the camera runs on an event-driven loop that only triggers a capture frame when motion is detected.

```mermaid
graph TD
    Start([Start]) --> Poll[Poll Motion Sensor State]
    Poll --> Detect{Is Motion Detected?}
    Detect -- No --> Poll
    Detect -- Yes --> Capture[Capture Frame]
    Capture --> Save[Save & Process Frame]
    Save --> Delay[Delay 1 second]
    Delay --> Poll
```

### 1.2 Image Analysis Pre-Filtering Algorithm

Before routing an image to a heavy Object Detection / Classification model, we run a lightweight filtering algorithm to discard unviable or static images.

1. **Format & Integrity Check:** Ensure the frame is fully loaded and file properties (e.g., height, width) are greater than zero.
2. **Exposure Check:** Analyze average pixel brightness. If the image is extreme (e.g., mean pixel value $< 10$ or $> 245$), mark as unreadable and terminate the pipeline.
3. **Frame Differencing:** Calculate the absolute pixel-by-pixel difference between the current frame and a cached reference frame (taken during a known static interval).
4. **Analysis Decision:**
   * If the absolute pixel variance exceeds a threshold limit (e.g., $>2\%$ of total pixels changed), flag the frame status as `Needs Analysis` and forward it to the inference step.
   * Otherwise, flag as `Static` and discard.

---

## 2. Testing with Pytest

### 2.1 Core Concepts
* **Test Isolation:** Pytest automatically runs functions prefixed with `test_` found in files named `test_*.py`.
* **Assert Statements:** Relies on standard Python assertions (`assert a == b`) rather than custom boilerplate methods.

### 2.2 Implementation Example
Below is an implementation of a detection filter and its corresponding Pytest test.

```python
# detection.py
def filter_detections(detections: list, threshold: float) -> list:
    """Filters out any detections that do not meet the minimum confidence threshold."""
    return [d for d in detections if d.get("confidence", 0.0) >= threshold]

# test_detection.py
from detection import filter_detections

def test_filter_detections_normal_case():
    sample_data = [
        {"label": "person", "confidence": 0.92},
        {"label": "dog", "confidence": 0.45},
        {"label": "car", "confidence": 0.12}
    ]
    # Filter anything below 0.50 confidence
    result = filter_detections(sample_data, 0.50)
    
    assert len(result) == 1
    assert result[0]["label"] == "person"

def test_filter_detections_empty_list():
    assert filter_detections([], 0.5) == []
```

## 3. TypeScript Transition
### 3.1 Core Concepts
Type Safety: Prevents runtime errors by checking object structures at compile time.
Union Types: Standardizes specific allowed states rather than using arbitrary string variables.
### 3.2 Transitioning JS Objects to TS Types
Below is a conversion mapping a dynamic Javascript pipeline state into strict TypeScript definitions.
```typescript
// 1. Define valid pipeline statuses
export type PipelineStatus = 'waiting' | 'running' | 'completed' | 'failed';

// 2. Define the structure of a bounding box coordinates
export type BoundingBox = [number, number, number, number]; // [x, y, width, height]

// 3. Define individual detection instances
export interface Detection {
  id: string;
  label: string;
  confidence: number;
  box: BoundingBox;
}

// 4. Define the primary pipeline state object
export interface PipelineState {
  currentStage: PipelineStatus;
  detections: Detection[];
  lastInferenceTimeMs: number | null;
  error?: string; // Optional field
}
```

## 4. Frontend Tooling with Vite
### 4.1 Core Concepts
ES Modules (ESM): Vite serves source code over native ESM, eliminating the need to rebundle the entire app on every file change.
HMR (Hot Module Replacement): Updates code in the browser instantly without losing current application state.

When a user clicks a button inside your live app and a piece of state changes, React uses its Virtual DOM to figure out exactly which HTML elements need to change. It updates only that specific part of the screen without refreshing the page.

Vite uses an incredibly fast engine called esbuild to instantly compile your JSX into standard JS the second you hit save.

For production, it takes hundreds of your separate code files, optimizes them, removes unused code (tree-shaking), and minifies them into a few highly compact files.

You cannot just double-click an index.html file on your computer to run a modern web app properly; security restrictions will block features like routing or fetching data. You need a local web server to "host" your files while you develop.

Vite spins up a lightweight, ultra-fast local server (usually http://localhost:5173).

HMR does not compare the DOM tree. Comparing the DOM tree is React's job.


## 5. State Management with Zustand
### 5.1 Core Concepts
No Boilerplate: Zustand does not require wrapping your React app in Context Providers.
Single Hook Store: State and state-mutating functions (actions) are kept together in a single custom React hook.
### 5.2 Implementation Example
This store controls the state of the image pipeline and allows components to subscribe to updates.

```typescript
import { create } from 'zustand';
import { PipelineStatus, Detection } from './types'; // Assuming types defined above

interface PipelineStore {
  status: PipelineStatus;
  detections: Detection[];
  startPipeline: () => void;
  setDetections: (newDetections: Detection[]) => void;
  resetPipeline: () => void;
}

export const usePipelineStore = create<PipelineStore>((set) => ({
  status: 'waiting',
  detections: [],
  
  startPipeline: () => set({ status: 'running' }),
  
  setDetections: (newDetections) => set({ 
    detections: newDetections, 
    status: 'completed' 
  }),
  
  resetPipeline: () => set({ 
    status: 'waiting', 
    detections: [] 
  })
}));
```

```typescript
import React from 'react';
import { usePipelineStore } from './store';

export const PipelineControlPanel: React.FC = () => {
  const status = usePipelineStore((state) => state.status);
  const startPipeline = usePipelineStore((state) => state.startPipeline);

  return (
    <div className="card">
      <h3>System Status: {status.toUpperCase()}</h3>
      <button 
        disabled={status === 'running'} 
        onClick={startPipeline}
      >
        Run Diagnostics
      </button>
    </div>
  );
};
```