"""
Server Module

This module defines the Server class representing computational servers
in the cloud environment that process tasks.

Each server has:
- CPU capacity: Maximum CPU resources available
- Task queue: Queue of tasks waiting to be processed
- Current CPU usage: Resources currently being used
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from collections import deque

# Import Task for type hints (avoid circular import with TYPE_CHECKING)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from env.task import Task


@dataclass
class Server:
    """
    Represents a computational server in the cloud environment.
    
    A server processes tasks from its queue, using CPU resources.
    Tasks are processed in FIFO order, and the server tracks its
    utilization and queue state.
    
    Attributes:
        server_id: Unique identifier for the server
        cpu_capacity: Maximum CPU capacity (e.g., 100 units)
        max_queue_length: Maximum tasks allowed in queue
        
    Internal State:
        task_queue: FIFO queue of tasks waiting to be processed
        current_tasks: List of tasks currently being processed
        current_cpu_usage: CPU resources currently in use
        total_tasks_processed: Counter of completed tasks
        total_latency: Sum of latencies for completed tasks
    
    Example:
        server = Server(server_id=0, cpu_capacity=100.0)
        server.add_task(task)
        server.step(current_time=5)
    """
    
    server_id: int
    cpu_capacity: float = 100.0
    max_queue_length: int = 20
    
    # Internal state (not passed to __init__)
    task_queue: deque = field(default_factory=deque, repr=False)
    current_tasks: List = field(default_factory=list, repr=False)
    current_cpu_usage: float = field(default=0.0, repr=False)
    total_tasks_processed: int = field(default=0, repr=False)
    total_latency: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        """Validate initialization values."""
        if self.cpu_capacity <= 0:
            raise ValueError("cpu_capacity must be greater than 0")
    
    @property
    def queue_length(self) -> int:
        """Get the current number of tasks in the queue."""
        return len(self.task_queue)
    
    @property
    def num_active_tasks(self) -> int:
        """Get the number of tasks currently being processed."""
        return len(self.current_tasks)
    
    @property
    def cpu_utilization(self) -> float:
        """
        Get the current CPU utilization as a fraction [0, 1].
        
        Returns:
            CPU usage divided by capacity, clamped to [0, 1]
        """
        return min(1.0, self.current_cpu_usage / self.cpu_capacity)
    
    @property
    def is_overloaded(self) -> bool:
        """
        Check if the server is overloaded.
        
        A server is considered overloaded if:
        - CPU utilization > 90% OR
        - Queue is at maximum capacity
        
        Returns:
            True if server is overloaded
        """
        return (self.cpu_utilization > 0.9 or 
                self.queue_length >= self.max_queue_length)
    
    @property
    def available_cpu(self) -> float:
        """Get the available CPU capacity."""
        return max(0.0, self.cpu_capacity - self.current_cpu_usage)
    
    @property
    def can_accept_task(self) -> bool:
        """Check if the server can accept a new task."""
        return self.queue_length < self.max_queue_length
    
    def add_task(self, task: 'Task') -> bool:
        """
        Add a task to the server's queue.
        
        Args:
            task: The task to add
            
        Returns:
            True if task was added, False if queue is full
        """
        if not self.can_accept_task:
            return False
        
        task.assign_to_server(self.server_id)
        self.task_queue.append(task)
        return True
    
    def _start_next_task(self, current_time: int) -> bool:
        """
        Start processing the next task from the queue if possible.
        
        Args:
            current_time: Current simulation time step
            
        Returns:
            True if a task was started, False otherwise
        """
        if len(self.task_queue) == 0:
            return False
        
        # Peek at the next task
        next_task = self.task_queue[0]
        
        # Check if we have enough CPU capacity
        if self.current_cpu_usage + next_task.cpu_requirement <= self.cpu_capacity:
            # Remove from queue and start processing
            task = self.task_queue.popleft()
            task.start(current_time)
            self.current_tasks.append(task)
            self.current_cpu_usage += task.cpu_requirement
            return True
        
        return False
    
    def step(self, current_time: int) -> List['Task']:
        """
        Advance the server by one time step.
        
        This method:
        1. Processes all active tasks
        2. Removes completed tasks
        3. Starts new tasks from queue if capacity available
        
        Args:
            current_time: Current simulation time step
            
        Returns:
            List of tasks that completed this step
        """
        completed_tasks = []
        tasks_to_remove = []
        
        # Process all active tasks
        for task in self.current_tasks:
            if task.process(current_time):
                # Task completed
                completed_tasks.append(task)
                tasks_to_remove.append(task)
                self.total_tasks_processed += 1
                
                # Track latency
                latency = task.get_latency()
                if latency is not None:
                    self.total_latency += latency
        
        # Remove completed tasks and free CPU
        for task in tasks_to_remove:
            self.current_tasks.remove(task)
            self.current_cpu_usage -= task.cpu_requirement
        
        # Ensure CPU usage doesn't go negative due to floating point errors
        self.current_cpu_usage = max(0.0, self.current_cpu_usage)
        
        # Try to start new tasks from queue
        while self._start_next_task(current_time):
            pass
        
        return completed_tasks
    
    def get_estimated_wait_time(self, task: 'Task') -> int:
        """
        Estimate the waiting time for a new task if added to this server.
        
        This is a rough estimate based on:
        - Current queue length
        - Average processing time of queued tasks
        - Current task processing progress
        
        Args:
            task: The task to estimate wait time for
            
        Returns:
            Estimated wait time in time steps
        """
        # Sum remaining time of current tasks
        active_remaining = sum(t.remaining_time for t in self.current_tasks)
        
        # Sum processing time of queued tasks
        queued_time = sum(t.processing_time for t in self.task_queue)
        
        # Simple estimate: assume sequential processing
        # In reality, tasks may process in parallel if CPU allows
        return active_remaining + queued_time
    
    def get_state(self) -> Tuple[float, int]:
        """
        Get the current state of the server for RL observation.
        
        Returns:
            Tuple of (cpu_utilization, queue_length)
        """
        return (self.cpu_utilization, self.queue_length)
    
    def get_average_latency(self) -> float:
        """
        Get the average latency of completed tasks.
        
        Returns:
            Average latency, or 0.0 if no tasks completed
        """
        if self.total_tasks_processed == 0:
            return 0.0
        return self.total_latency / self.total_tasks_processed
    
    def reset(self) -> None:
        """
        Reset the server to its initial state.
        
        Clears all tasks and resets counters for a new episode.
        """
        self.task_queue.clear()
        self.current_tasks.clear()
        self.current_cpu_usage = 0.0
        self.total_tasks_processed = 0
        self.total_latency = 0
    
    def get_info(self) -> dict:
        """
        Get detailed information about the server state.
        
        Returns:
            Dictionary with server statistics
        """
        return {
            'server_id': self.server_id,
            'cpu_utilization': self.cpu_utilization,
            'queue_length': self.queue_length,
            'active_tasks': self.num_active_tasks,
            'is_overloaded': self.is_overloaded,
            'total_processed': self.total_tasks_processed,
            'avg_latency': self.get_average_latency()
        }
    
    def __repr__(self) -> str:
        """String representation of the server."""
        return (f"Server(id={self.server_id}, "
                f"cpu={self.cpu_utilization:.1%}, "
                f"queue={self.queue_length})")
