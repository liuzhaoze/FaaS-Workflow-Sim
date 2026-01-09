# Serverless 强化学习环境

本强化学习环境包含 `Function`、`Workflow`、`Workload`、`NumaNode`、`Server`、`Cluster` 共六个实体，模拟了 serverless 工作流在异构 NUMA 集群中的执行过程。

## Function

`Function` 用于建模构成 serverless 工作流的云函数，定义在 `function.py` 文件中。`Function` 包含以下属性：

- `wf_id`: 该函数所属工作流的 ID
- `fn_id`: 该函数在工作流中的 ID
- `computation`: 执行该函数所需的 CPU 计算操作数量
- `memory_req`: 执行该函数所需的内存大小
- `parallelism`: 该函数的并行度
- `state`: 该函数的状态，包括四种：
  - `PENDING`: 函数的初始状态
  - `SUBMITTED`: 函数已经提交到执行队列中
  - `RUNNING`: 函数正在集群中的某个 NUMA 节点上执行
  - `FINISHED`: 函数已经执行完成
- `memory_alloc`: 分配给该函数的内存大小
- `submission_time`: 函数被提交到执行队列的时间
- `start_time`: 函数在 NUMA 节点上开始执行的时间
- `completion_time`: 函数执行完成的时间

`Function` 的状态转移过程如下：

1. `PENDING` $\rightarrow$ `SUBMITTED`:
   - 如果该函数在工作流中是源点（没有前驱函数），那么它的提交时间就是工作流执行请求到达的时间
   - 如果该函数有前驱函数，那么它的提交时间是其所有前驱函数完成时间的最大值
2. `SUBMITTED` $\rightarrow$ `RUNNING`:
   - 调度器做出将该函数分配到某个 NUMA 节点的决策后，首先要经历可能存在的冷启动延迟 `cold_start_latency`，然后要经历前驱函数向该函数传输数据的时间 `data_transfer_time`，然后才能开始在 NUMA 节点上执行
3. `RUNNING` $\rightarrow$ `FINISHED`:
   - 函数在 NUMA 节点上执行结束

## Workflow

`Workflow` 用于建模 serverless 工作流，定义在 `workflow.py` 文件中。`Workflow` 包含以下属性：

- `wf_id`: 工作流的 ID
- `arrival_time`: 工作流执行请求到达的时间
- `_functions`: 储存了工作流中所有函数实例
- `_dag`: 储存了工作流的 DAG 结构
- `source_functions`: 工作流中的源点函数集合
- `pending_functions`: 工作流中处于 `PENDING` 状态的函数集合
- `submitted_functions`: 工作流中处于 `SUBMITTED` 状态的函数集合
- `running_functions`: 工作流中处于 `RUNNING` 状态的函数集合
- `finished_functions`: 工作流中处于 `FINISHED` 状态的函数集合

## Workload

`Workload` 是 serverless 系统的工作负载，它包含了所有要执行的工作流，并且可以通过它获取下一个要执行的函数，定义在 `workload.py` 文件中。

首先调用 `Workload.reset()` 方法初始化工作负载。这个函数会重置所有工作流的状态，并且把所有工作流的源点函数都添加到提交队列中。

然后调用 `Workload.get()` 方法源源不断地获取下一个要执行的函数。

获取到的函数会根据调度器的决策被分配到某个 NUMA 节点上执行。此时使用 `Workload.run_function(...)` 方法更新工作负载中特定工作流的函数的状态。

当函数执行完成后，使用 `Workload.finish_function(...)` 方法更新工作负载中特定工作流的函数的状态。因为函数的完成可能会使得它的后继函数变为可提交状态，所以 `Workload.finish_function` 还需要检查已完成的函数的所有后继函数，看看这些后继函数的前驱函数是否都已经完成，如果是的话，就把这些后继函数添加到提交队列中。

## Container

`Container` 是在 NUMA 节点上执行的云函数容器，定义在 `container.py` 文件中。`Container` 包含以下属性：

- `wf_id`: 该容器所执行的函数所属工作流的 ID
- `fn_id`: 该容器所执行的函数在工作流中的 ID
- `memory_req`: 该容器所执行的函数所需的内存大小 (MB)
- `memory_alloc`: 分配给该容器的内存大小 (MB)
- `computation`: 该容器所执行的函数所需的 CPU 计算操作数量
- `parallelism`: 该容器所执行的函数的并行度
- `submission_time`: 该容器所执行的函数被提交到执行队列的时间
- `data_transfer_time`: 传输前驱函数数据到该容器所需的时间
- `remaining_computation`: 该容器中的函数剩余的计算操作数量
- `creation_time`: 该容器在 NUMA 节点上创建的时间
- `start_time`: 该容器在 NUMA 节点上开始执行的时间
- `completion_time`: 该容器执行完成的时间

## NumaNode

`NumaNode` 是异构服务器集群中服务器上的 NUMA 节点，定义在 `numa_node.py` 文件中。`NumaNode` 是真正用来执行云函数的实体。`NumaNode` 包含以下属性：

- `node_id`: NUMA 节点的 ID
- `cpu`: NUMA 节点的 CPU 核心数
- `memory`: NUMA 节点的内存大小
- `single_core_speed`: CPU 单核计算速度 (单个 CPU 核心每秒可执行的计算操作数量)
- `computing_capacity`: NUMA 节点的计算能力（等于 `cpu` $\times$ `single_core_speed` ）
- `_waiting_containers`: 等待创建的容器队列（当 NUMA 节点内存不足时，容器无法立即创建，就会进入等待队列，直到有足够的内存可以分配给容器）
- `_running_containers`: 正在运行的容器列表
- `_current_time`: NUMA 节点的当前时间（只有在容器完成时才会更新；因为容器创建可能会影响当前已经计算好的容器执行速度和完成时间，需要在容器创建后重新计算自当前时间起每个正在运行的容器的执行速度和完成时间；而容器完成不会影响已经计算好的执行速度和完成时间，因此可以向前推进当前时间）
- `_rum`: [`ResourceUtilizationManager`](#resourceutilizationmanager) 用于记录 NUMA 节点的 CPU 利用率、内存利用率、负载水平、总并行度和空闲内存的时间序列数据
- `_esm`: [`ExecutionSpeedManager`](#executionspeedmanager) 用于记录每个容器在 NUMA 节点上的执行速度的变化情况（这是用于更新容器剩余计算量的重要依据：计算剩余计算量，需要知道已经完成的计算量）

### NumaNode 方法调用流程

1. 首先使用 `reset` 方法初始化 NUMA 节点的状态
2. 然后调用 `on_container_creation` 方法来执行容器创建时的操作
   1. 如果在容器创建时，NUMA 节点上没有足够的内存分配给容器，那么就将创建请求添加到等待队列中，并且只有在 NUMA 节点上有容器执行完成时，才会检查等待队列中的容器创建请求，看看是否有足够的内存来创建这些容器（因为只有容器完成才会释放内存）
   2. 如果在容器创建时，NUMA 节点上有足够的内存分配给容器，那么就直接将该容器添加到正在运行的容器列表中
   3. 因为创建容器运行会导致 NUMA 节点上总并行度的增加，从而影响其他容器的执行速度，进而影响它们的完成时间，所以需要在容器创建后调用 [`_update_completion_times`](#_update_completion_times) 方法，重新计算自当前时间起每个正在运行的容器的执行速度和完成时间，并且更新资源利用率记录和执行速度记录
3. 接着根据实际需求调用 `on_container_completion` 方法来处理容器真正完成时的操作（既然要完成一个容器，肯定是完成结束时间最早的那个容器）
   1. 首先将最早完成的容器从正在运行的容器列表中移除
   2. 然后调用 [`_update_remaining_computations`](#_update_remaining_computations) 方法，更新这个容器完成时，其他正在运行的容器的剩余计算量
   3. 将当前时间推进到该容器的完成时间（这里就是真实地将时间推进到下一个容器完成的时间，而不是模拟。这是因为调用 `on_container_completion` 时，必须确信这个容器完成之前不会有新的容器创建请求到来，NUMA 节点中各容器的状态截止到该容器完成时都不会再发生变化）
   4. 因为该容器完成了会释放内存，所以那些之前因为内存不足被放在等待队列中的容器创建请求现在可能可以被创建了。所以使用 [`_run_waiting_containers`](#_run_waiting_containers) 方法从等待队列中取出尽可能多的容器创建请求来创建容器
   5. 因为有容器被创建了，新容器加入到了正在运行的容器列表中，所以最后还要调用 [`_update_completion_times`](#_update_completion_times) 方法，重新计算自当前时间起（即该容器执行结束，并且新容器从等待队列进入正在运行容器列表后）每个正在运行的容器的执行速度和完成时间
   6. 最后返回该容器，以记录该容器的各项时间点

### NumaNode 辅助函数

#### _update_completion_times

该函数更新自当前时间起每个正在运行的容器的执行速度和完成时间。函数首先使用一个局部变量 `elapsed_time` 作为时间指针，在模拟容器执行过程中用于记录时间的推进。然后清除当前时间及之后的资源利用率和所有的容器执行速度记录，为了在后面模拟过程中添加新的记录。（这样操作可以保证执行记录总是从当前时间开始，因此方便在 `on_container_completion` 中计算除已完成容器之外的容器的剩余计算量。）

然后函数开始模拟容器在 NUMA 节点上的执行过程。之所以说是模拟，是因为该函数计算的完成时间有可能不是实际完成时间。这是因为后续可能会有新的容器创建请求通过 `on_container_creation` 方法添加到 NUMA 节点上，从而影响当前已经计算好的容器的执行速度和完成时间。

函数用一个名为 `active_containers` 的字典记录 `elapsed_time` 时刻正在运行的容器的剩余计算量，用一个名为 `future_containers` 的列表按时间先后顺序记录所有后续要开始执行的容器。并用 `total_parallelism` 和 `used_memory` 分别记录 `elapsed_time` 时刻的总并行度和已使用内存。

在模拟过程中，只有两种事件会影响容器的执行速度，进而影响容器的完成时间：容器开始事件和容器结束事件。当容器开始执行时，NUMA 节点上的总并行度会增加。在总并行度超过 CPU 核心数的情况下，每个并行度的执行速度会下降，从而影响所有正在运行的容器的执行速度和完成时间。当容器执行完成时，NUMA 节点上的总并行度会减少，从而使得每个并行度的执行速度上升，进而影响所有正在运行的容器的执行速度和完成时间。同时，这两个事件的发生也意味着 NUMA 节点的资源利用率发生了变化，因此也需要记录资源利用率的变化情况。

综上，使用一个循环按照时间顺序处理所有容器的开始和结束事件。每遇到一个事件，就先计算 `elapsed_time` 到该事件发生时间内所有活动容器的已完成计算量，然后更新剩余计算量，接着推进时间指针 `elapsed_time` 到该事件发生时间，最后根据事件类型更新活动容器列表、总并行度和已使用内存。接着在下一个循环中计算此时刻所有活动容器的执行速度和完成时间，直到所有容器的开始和结束事件都被处理完为止。

##### 容器执行速度模型

下面通过一个例子来说明容器的执行速度是如何计算的：

![container_execution_model](./images/container_execution_model.drawio.svg)

假设一个 NUMA 节点有 4 个 CPU 核心，8192 MB 内存，单核计算速度为 1000 ，总计算能力为 4000 。

在 `NumaNode._current_time` 时刻，NUMA 节点上有两个容器 A 和 B 在运行（即：这两个容器的 `start_time` $\leq$ `NumaNode._current_time`）。假设容器 A 的并行度为 1 ，需要 512 MB 内存，并且给容器 A 分配了 512 MB 内存，剩余计算量为 2585 ；容器 B 的并行度为 2 ，需要 1024 MB 内存，并且给容器 B 分配了 1536 MB 内存，剩余计算量为 7171 。此时 NUMA 节点的总并行度为 3 ，小于 CPU 核心数 4 ，所以单个并行度的执行速度为 1000 。因此容器 A 的执行速度为 1000 ，容器 B 的执行速度为 2000 ，NUMA 节点的 CPU 利用率为 (1000 + 2000) / 4000 = 75% 。容器 A 预计在 2.585 秒后（即 `NumaNode._current_time + 2.585` 时刻）执行完成，容器 B 预计在 3.5855 秒后（即 `NumaNode._current_time + 3.5855` 时刻）执行完成。

假设 1 秒之后（即 `NumaNode._current_time + 1.0` 时刻），有一个新容器 C 开始执行，容器 C 的并行度为 4 ，需要 1024 MB 内存，并且给容器 C 分配了 1024 MB 内存，剩余计算量为 1142 。那么 NUMA 节点的总并行度变为 7 ，大于 CPU 核心数 4 。根据时间片轮转的思想，每个并行度分到的计算速度为 $\left\lfloor 4000 / 7 \right\rfloor = 571$ （计算操作数量一定是整数，因此这里向下取整）。因此容器 A 的执行速度变为 $1 \times 571 = 571$ ，容器 B 的执行速度变为 $2 \times 571 = 1142$ ，容器 C 的执行速度变为 $4 \times 571 = 2284$ ，NUMA 节点的 CPU 利用率变为 (571 + 1142 + 2284) / 4000 = 99.925% 。此时容器 A 的剩余计算量为 $2585 - \left\lfloor 1000 \times 1.0 \right\rfloor = 1585$ ，容器 B 的剩余计算量为 $7171 - \left\lfloor 2000 \times 1.0 \right\rfloor = 5171$ 。因此容器 A 预计在 $1585 / 571 \approx 2.776$ 秒后（即 `NumaNode._current_time + 3.776` 时刻）执行完成，容器 B 预计在 $5171 / 1142 \approx 4.524$ 秒后（即 `NumaNode._current_time + 5.524` 时刻）执行完成，容器 C 预计在 $1142 / 2284 \approx 0.5$ 秒后（即 `NumaNode._current_time + 1.5` 时刻）执行完成。

假设 0.5 秒内（即 `NumaNode._current_time + 1.5` 时刻之前）没有新创建的容器，那么容器 C 首先执行完成。那么 NUMA 节点的总并行度变回 3 ，因此容器 A 的执行速度变回 1000 ，容器 B 的执行速度变回 2000 ，NUMA 节点的 CPU 利用率变回 75% 。此时容器 A 的剩余计算量为 $1585 - \left\lfloor 571 \times 0.5 \right\rfloor = 1300$ ，容器 B 的剩余计算量为 $5171 - \left\lfloor 1142 \times 0.5 \right\rfloor = 4600$ 。因此容器 A 预计在 $1300 / 1000 = 1.3$ 秒后（即 `NumaNode._current_time + 2.8` 时刻）执行完成，容器 B 预计在 $4600 / 2000 = 2.3$ 秒后（即 `NumaNode._current_time + 3.8` 时刻）执行完成。（注意：预计执行时间比之前计算的时间变短了，因为容器 C 执行完成后，NUMA 节点的总并行度减少了，单个并行度的执行速度上升了。）

又过了 0.5 秒后（即 `NumaNode._current_time + 2.0` 时刻），有一个新容器 D 开始执行，容器 D 的并行度为 2 ，需要 2048 MB 内存，但是给容器 D 分配了 1536 MB 内存（内存不足），剩余计算量为 4200 。此时 NUMA 节点的总并行度变为 5 ，大于 CPU 核心数 4 。每个并行度分到的计算速度为 $\left\lfloor 4000 / 5 \right\rfloor = 800$ 。因此容器 A 的执行速度变为 $1 \times 800 = 800$ ，容器 B 的执行速度变为 $2 \times 800 = 1600$ 。为了模拟内存不足导致频繁的内存交换所带来的性能下降，容器 D 的执行速度为 $\left\lfloor 1536 / 2048 \times 2 \times 800 \right\rfloor = 1200$ 。此时 NUMA 节点的 CPU 利用率为 (800 + 1600 + 1200) / 4000 = 90% 。此时容器 A 的剩余计算量为 $1300 - \left\lfloor 1000 \times 0.5 \right\rfloor = 800$ ，容器 B 的剩余计算量为 $4600 - \left\lfloor 2000 \times 0.5 \right\rfloor = 3600$ 。因此容器 A 预计在 $800 / 800 = 1.0$ 秒后（即 `NumaNode._current_time + 3.0` 时刻）执行完成，容器 B 预计在 $3600 / 1600 = 2.25$ 秒后（即 `NumaNode._current_time + 4.25` 时刻）执行完成，容器 D 预计在 $4200 / 1200 = 3.5$ 秒后（即 `NumaNode._current_time + 5.5` 时刻）执行完成。（因为总并行度增加了，单个并行度的执行速度下降了，所以容器 A 和 B 的预计执行时间变长了。）

假设后续没有新创建的容器，那么过了 1 秒后（即 `NumaNode._current_time + 3.0` 时刻），容器 A 率先执行完成。NUMA 节点的总并行度变为 4 ，没有超过 CPU 核心数 4 ，那么单个并行度的执行速度变回 1000 。因此容器 B 的执行速度变为 $2 \times 1000 = 2000$ ，容器 D 的执行速度为 $\left\lfloor 1536 / 2048 \times 2 \times 1000 \right\rfloor = 1500$ ，NUMA 节点的 CPU 利用率变为 (2000 + 1500) / 4000 = 87.5% 。此时容器 B 的剩余计算量为 $3600 - \left\lfloor 1600 \times 1.0 \right\rfloor = 2000$ ，容器 D 的剩余计算量为 $4200 - \left\lfloor 1200 \times 1.0 \right\rfloor = 3000$ 。因此容器 B 预计在 $2000 / 2000 = 1.0$ 秒后（即 `NumaNode._current_time + 4.0` 时刻）执行完成，容器 D 预计在 $3000 / 1500 = 2.0$ 秒后（即 `NumaNode._current_time + 5.0` 时刻）执行完成。

接下来 1 秒后（即 `NumaNode._current_time + 4.0` 时刻），容器 B 执行完成。NUMA 节点的总并行度变为 2 ，单个并行度的执行速度不变，容器 D 的执行速度不变，NUMA 节点的 CPU 利用率变为 1500 / 4000 = 37.5% 。此时容器 D 的剩余计算量为 $3000 - \left\lfloor 1500 \times 1.0 \right\rfloor = 1500$ 。因此容器 D 预计在 $1500 / 1500 = 1.0$ 秒后（即 `NumaNode._current_time + 5.0` 时刻）执行完成。

最后 1 秒后（即 `NumaNode._current_time + 5.0` 时刻），容器 D 执行完成，NUMA 节点上没有正在运行的容器了。

#### _update_remaining_computations

该函数只在 `NumaNode.on_container_completion` 方法中被调用。主调方法的逻辑明确了 `NumaNode._current_time` 只会随着容器的完成而推进。而计算容器的剩余计算量依据的是从上一个容器完成到当前容器完成这段时间内各个容器完成的计算量。而 `ExecutionSpeedManager` 的记录正好都是以当前时间（即上一个容器完成时间）为起点的，因此 `ExecutionSpeedManager.get_completed_computation` 只需要指定截止的时间点就能获得上一个容器完成到当前容器完成这段时间内某个容器已经完成的计算量。据此更新当下一个容器完成时，其他正在运行的容器的剩余计算量。

#### _run_waiting_containers

该函数同样只在 `NumaNode.on_container_completion` 方法中被调用。函数会先从 `ResourceUtilizationManager` 中获得指定时间的空闲内存，然后不断地从等待队列中取出容器创建请求，放入正在运行的容器列表中，直到没有足够的内存来创建下一个容器为止。

### NumaNode 辅助类

#### ResourceUtilizationManager

`ResourceUtilizationManager` 中的每条记录包含如下字段：

- `timestamp`: 记录的时间点
- `cpu`: 该时间点及之后的 CPU 利用率
- `memory`: 该时间点及之后的内存利用率
- `load`: 该时间点及之后的负载水平
- `total_parallelism`: 该时间点及之后的总并行度
- `free_memory`: 该时间点及之后的空闲内存 (MB)

这些记录按照 `timestamp` 字段升序排列。它们表示自该时间点起，直到下一条记录时间点为止的资源利用率情况。

> **例 1**
>
> 假设有三条记录，它们的 `timestamp` 为：`[0.0, 1.0, 1.5, 3.0]`。这就表示从 0.0 秒到 1.0 秒这段时间内，NUMA 节点的资源利用率是第一条记录的内容；从 1.0 秒到 1.5 秒这段时间内，NUMA 节点的资源利用率是第二条记录的内容；从 1.5 秒到 3.0 秒这段时间内，NUMA 节点的资源利用率是第三条记录的内容；从 3.0 秒开始，NUMA 节点的资源利用率始终是最后一条记录的内容。
>
> **例 2**
>
> 如果只有一条记录，即 `timestamp` 为 `[0.0]`，那么这就表示从 0.0 秒开始，NUMA 节点的资源利用率始终是这一条记录的内容。

使用方法：

1. 首先使用 `reset` 方法初始化资源利用率记录，该方法会清空之前的记录，并且添加一条初始记录，表示 NUMA 节点的初始资源利用率情况
2. 在添加新记录之前，需要使用 `clear_after` 方法将当前时间及之后的旧纪录清除掉
3. 不断使用 `add_record` 方法添加新的资源利用率记录

#### ExecutionSpeedManager

`ExecutionSpeedManager` 记录了 `NumaNode._running_containers` 中每个容器的执行速度变化情况。它使用一个字典来存储每个容器的执行速度记录列表，字典的键是容器对应的工作流ID和函数ID的元组 `(wf_id, fn_id)`，值是一个执行速度记录列表。列表中的每条记录包含如下字段：

- `timestamp`: 记录的时间点
- `speed`: 该时间点及之后的执行速度

这些记录按照 `timestamp` 字段升序排列。它们表示自该时间点起，直到下一条记录时间点为止的执行速度。它们的含义与 [`ResourceUtilizationManager`](#resourceutilizationmanager) 中示例所述的类似。

使用方法：

1. 首先使用 `reset` 方法初始化执行速度记录，该方法会清空之前的记录
2. 然后就可以不断使用 `add_record` 方法为特定容器添加新的执行速度记录
3. 使用 `get_completed_computation` 方法可以获得从第一条记录的时间点起，到指定时间点为止，该容器已经完成的计算量（相当于执行速度乘以时间的积分）

## Server

`Server` 是拥有一个或多个 NUMA 节点的服务器，定义在 `server.py` 文件中。`Server` 包含以下属性：

- `server_id`: 属于该类型的服务器的 ID
- `hourly_rate`: 服务器每小时的价格
- `cold_start_latency`: 服务器冷启动的时间
- `_numa_nodes`: 服务器上的 NUMA 节点集合
- `expiration_time`: 服务器的到期时间
- `earliest_finished_time`: 服务器上所有 NUMA 节点中最早完成的容器的完成时间
- `earliest_finished_nn_id`: 服务器上所有 NUMA 节点中最早完成的容器所在的 NUMA 节点 ID
- `latest_finished_time`: 服务器上所有 NUMA 节点中最晚完成的容器的完成时间
- `latest_finished_nn_id`: 服务器上所有 NUMA 节点中最晚完成的容器所在的 NUMA 节点 ID
- `_container_count`: 服务器上已经创建但未完成的容器总数

### Server 方法调用流程

1. 首先调用 `reset` 方法初始化服务器的状态
2. 在服务器的指定 NUMA 节点上创建容器之前，先使用 `startup_at` 方法启动服务器并获得冷启动延迟
3. 然后调用 `on_container_creation` 方法在指定的 NUMA 节点上创建容器
   1. 首先调用该 NUMA 节点的 `on_container_creation` 方法来执行容器创建时的操作
   2. 接着调用 [`_update_earliest_and_latest_finished`](#_update_earliest_and_latest_finished) 方法，用来更新服务器记录的拥有最早和最晚完成的容器的 NUMA 节点以及对应的完成时间
   3. 然后调用 [`_renew_lease_at`](#_renew_lease_at) 方法执行租用服务器的操作
4. 最后根据实际需求调用 `on_container_completion` 方法来执行最早完成的容器实际完成时的操作
   1. 首先调用该 NUMA 节点的 `on_container_completion` 方法来执行容器真正完成时的操作
   2. 虽然容器真正完成不会影响已经计算好的完成时间，但是容器完成后可能会将等待队列中的容器取出来创建，此时就有可能影响已经计算好的完成时间，所以仍然需要调用 [`_update_earliest_and_latest_finished`](#_update_earliest_and_latest_finished) 方法来更新服务器记录的拥有最早和最晚完成的容器的 NUMA 节点以及对应的完成时间
   3. 同样处于上述原因，也需要调用 [`_renew_lease_at`](#_renew_lease_at) 方法执行租用服务器的操作

### Server 辅助变量

#### earliest_finished_nn_id 与 earliest_finished_time

这两个变量用于记录服务器上最早完成的容器的完成时间和所在的 NUMA 节点 ID 。在 `on_container_completion` 方法中使用该变量确定调用哪个 NUMA 节点的 `on_container_completion` 方法。

#### latest_finished_nn_id 与 latest_finished_time

这两个变量用于记录服务器上最晚完成的容器的完成时间和所在的 NUMA 节点 ID 。在 [_renew_lease_at](#_renew_lease_at) 方法中使用该变量来确定服务器租期需要延长到什么时候，以覆盖所有正在运行的容器的执行。

### Server 辅助函数

#### _update_earliest_and_latest_finished

该函数用于更新 [`earliest_finished_nn_id` / `earliest_finished_time`](#earliest_finished_nn_id-与-earliest_finished_time) 和 [`latest_finished_nn_id` / `latest_finished_time`](#latest_finished_nn_id-与-latest_finished_time) 这四个变量。

容器的完成不会影响已经计算好的完成时间；只有当有新容器运行时，需要重新计算完成时间。`on_container_creation` 和 `on_container_completion` 方法都有可能运行新的容器（前者在创建完容器后，如果内存充足就直接运行该容器；后者在完成容器后，如果内存充足就会从等待队列中运行新的容器），因此需要在这两个方法调用 `NumaNode.on_container_creation` 和 `NumaNode.on_container_completion` 方法后调用本方法更新上述四个变量。

下面以更新 `earliest_finished_nn_id` / `earliest_finished_time` 为例说明更新逻辑，`latest_finished_nn_id` / `latest_finished_time` 的更新逻辑类似。变量的更新分为如下情况：

1. 如果当前节点是最早完成节点，但是该节点的最早完成时间变晚了，那么就意味着该节点有可能不再是整个服务器的最早完成节点了，因此需要遍历所有 NUMA 节点来找出真正的最早完成节点
2. 如果当前节点不是最早完成节点，那么即使该节点的最早完成时间变晚了，也不会影响整个服务器的最早完成节点和时间的正确性，因此不需要做任何操作
3. 如果当前节点的最早完成时间比整个服务器的最早完成时间更早，那么无论整个服务器的最早完成节点是不是当前节点，都需要将最早完成节点和时间更新为当前节点和其对应的时间

体现在代码中，情况 2 和情况 3 可以放到一起处理（使用 `_update_earliest_compare` ），情况 1 单独处理（使用 `_update_earliest_scan` ）。

上述逻辑还有一个好处：当整个服务器上的最后一个容器完成后，最早完成时间应该设置为正无穷。如果使用 `_update_earliest_compare` 的逻辑，服务器的最早完成时间是有穷值，正无穷永远不会小于它，因此无法更新为正无穷。但是注意到：当服务器只剩下最后一个容器时，服务器记录的最早完成节点肯定是该容器所在的节点，而最后一个容器完成后，该节点的最早完成时间变为正无穷（也就是变晚了），正好符合情况 1 的条件，`_update_earliest_scan` 的逻辑正好可以处理所有 NUMA 节点都没有正在运行的容器的情况，从而正确地将服务器的最早完成时间更新为正无穷。

#### _renew_lease_at

该函数用于延长服务器的租期，以覆盖所有正在运行的容器的执行。

在 `on_container_creation` 方法中，向本方法传入的时间是函数的提交时间而非容器的创建时间。这是因为如果是新租服务器，服务器的租期应该从函数提交时间开始算起，然后是冷启动、数据传输，最后创建容器并执行。这些时间都应该被服务器租期覆盖。如果是续租服务器，那么服务器的租期应该从到期时间之后开始累加。

在 `on_container_completion` 方法中，向本方法传入的时间是容器的完成时间，因为容器完成时服务器要么处于租赁状态，要么恰好要到期。无论是哪种情况，容器的完成时间都不会超过服务器的到期时间，因此租期都会从到期时间之后开始累加。

## Cluster

`Cluster` 包含了所有异构服务器，定义在 `cluster.py` 文件中。`Cluster` 包含以下属性：

- `memory_bandwidth`: 单个 NUMA 节点的内存带宽 (bytes/second)
- `numa_bandwidth`: NUMA 节点之间的内存总线带宽 (bytes/second)
- `network_bandwidth`: 服务器之间的网络带宽 (bytes/second)
- `single_core_speed`: CPU 单核计算速度 (单个 CPU 核心每秒可执行的计算操作数量)
- `_servers`: 集群中的所有服务器（以字典形式储存，键是服务器的类型名称，值是该类型的服务器元组）
- `earliest_finished_srv_name`: 集群中所有服务器中最早完成的容器所在的服务器类型名称
- `earliest_finished_srv_id`: 集群中所有服务器中最早完成的容器所在的服务器 ID
- `earliest_finished_time`: 集群中所有服务器中最早完成的容器的完成时间
- `latest_finished_srv_name`: 集群中所有服务器中最晚完成的容器所在的服务器类型名称
- `latest_finished_srv_id`: 集群中所有服务器中最晚完成的容器所在的服务器 ID
- `latest_finished_time`: 集群中所有服务器中最晚完成的容器的完成时间

### Cluster 方法调用流程

详见 `RawEnv` 中 [step 方法执行流程](#step-方法执行流程) 部分。

## RawEnv

`RawEnv` 描述了 Serverless 强化学习环境的状态转移机制，定义在 `raw_env.py` 文件中。其使用方法如下：

1. 使用 `arrival_times` 、`workflow_templates` 和 `cluster_config` 初始化 `RawEnv` 实例
2. 使用 `reset` 方法初始化环境状态
3. 不断调用 `step` 方法为每个工作流中的函数执行分配资源，直到工作负载执行完成

### step 方法执行流程

1. 接收到资源分配决策后，记录执行该函数容器的服务器类型名称、服务器 ID 和 NUMA 节点 ID
2. 使用 `Cluster.start_server` 方法在当前函数提交时间点启动目标服务器，并获得冷启动延迟
   1. 该方法调用 `Server.startup_at` 在函数提交时间点启动服务器；如果函数提交时服务器已经启动，则不会有冷启动延迟，返回 0 ；否则返回服务器的冷启动时间
   2. `Server.startup_at` 还会对每个 NUMA 节点调用 `NumaNode.on_server_startup` 来设置 `NumaNode._current_time` 为启动后的时间点
3. 通过 `Cluster.get_data_transfer_speed` 方法获取当前函数与所有前驱函数所在的 NUMA 节点之间的数据传输速度，并计算数据传输时间
4. 服务器冷启动和数据传输时间计算完成后，创建容器实例
   - 处理当前函数时，不需要考虑当前的冷启动时间和数据传输时间内是否有其他函数执行完成导致其后继函数被提交的情况，因为资源分配的决策实在函数提交时做出的，即使后继函数被提交，也不会影响当前函数资源分配决策的处理
5. 调用 `Cluster.on_container_creation` 方法在目标服务器的指定 NUMA 节点上创建容器，同时累加租金
6. 获取下一个待决策的函数，需要考虑以下两种情况：
   1. 当前函数提交队列为空：
      - 工作负载中的所有函数都已经被分配了资源，但是还有容器没有执行完成；此时只需要不断调用 `Cluster.on_container_completion` 方法执行完所有容器即可
      - 工作流中有一个函数的所有前驱函数没有全部执行完成，因此该函数无法被提交；此时需要不断调用 `Cluster.on_container_completion` 方法，直到该函数的所有前驱函数都执行完成，从而使该函数被提交
   2. 当前函数提交队列不为空：
      - 需要先完成那些早于下一个待决策函数提交时间完成的容器，因为这些容器的完成可能会使其后继函数被提交，这些后继函数的提交时间早于下一个待决策函数的提交时间，所以真正的下一个待决策函数是这些后继函数
