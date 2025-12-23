#!/bin/bash

# ============================================================================
# 训练管理脚本
# 功能：启动后台训练、管理训练进程、查看日志
# 使用配置文件管理训练参数
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PID_DIR="${SCRIPT_DIR}/pids"
LOG_DIR="${SCRIPT_DIR}/log"
CFG_DIR="${SCRIPT_DIR}/cfg_config"
OUTPUT_BASE_DIR="${SCRIPT_DIR}/output"
TRAIN_SCRIPT="${PROJECT_ROOT}/scripts/train_model.py"

# 确保必要的目录存在
mkdir -p "${PID_DIR}"
mkdir -p "${LOG_DIR}"
mkdir -p "${CFG_DIR}"
mkdir -p "${OUTPUT_BASE_DIR}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ============================================================================
# 函数：显示使用帮助
# ============================================================================
show_help() {
    cat << EOF
训练管理脚本使用说明

用法:
    $0 <command> [options]

命令:
    start <config_file>      启动训练任务（使用配置文件，后台运行）
    stop [job_name]          停止训练任务（默认停止所有）
    status [job_name]        查看训练任务状态（默认查看所有）
    list                     列出所有训练任务
    logs <job_name>          查看训练日志（实时跟踪）
    configs                  列出所有可用的配置文件
    help                     显示此帮助信息

示例:
    # 启动一个训练任务（使用配置文件）
    $0 start cfg_config/my_experiment.json

    # 查看所有任务状态
    $0 status

    # 停止特定任务
    $0 stop my_experiment

    # 查看日志
    $0 logs my_experiment

    # 查看所有配置文件
    $0 configs

注意:
    - 配置文件保存在 ./cfg_config/ 目录下
    - 任务名称从配置文件名提取（不含扩展名）
    - 配置文件必须是有效的JSON格式
    - 日志文件保存在 ./log/ 目录下
    - PID文件保存在 ./pids/ 目录下
    - 输出文件保存在 ./output/{dataset}-{model}-{name}/ 目录下

配置文件格式:
    配置文件应为JSON格式，包含所有训练参数，例如:
    {
        "name": "my_experiment",
        "model": "resnet50",
        "dataset": "nih",
        "num_epochs": 100,
        "batch_size": 32,
        "lr": 0.001,
        ...
    }
EOF
}

# ============================================================================
# 函数：从配置文件路径提取任务名称
# ============================================================================
extract_job_name_from_config() {
    local config_file="$1"
    local basename=$(basename "$config_file")
    local name="${basename%.*}"  # 移除扩展名
    
    # 如果配置文件中指定了name，优先使用配置文件的name
    if command -v python3 &> /dev/null; then
        local cfg_name=$(python3 -c "
import json
import sys
try:
    with open('$config_file', 'r') as f:
        cfg = json.load(f)
        if 'name' in cfg:
            print(cfg['name'])
except:
    pass
" 2>/dev/null)
        if [[ -n "$cfg_name" ]]; then
            echo "$cfg_name"
            return
        fi
    fi
    
    echo "$name"
}

# ============================================================================
# 函数：将JSON配置文件转换为命令行参数
# ============================================================================
config_to_args() {
    local config_file="$1"
    local output_dir="$2"  # 输出目录路径（由调用者提供）
    
    if [[ ! -f "$config_file" ]]; then
        echo -e "${RED}错误: 配置文件不存在: ${config_file}${NC}" >&2
        return 1
    fi
    
    # 使用Python解析JSON并转换为命令行参数
    # 注意：由于argparse的type=bool缺陷，我们使用特殊标记来区分True/False
    python3 << EOF
import json
import sys
import os

try:
    with open('$config_file', 'r') as f:
        config = json.load(f)
    
    args = []
    
    # 定义参数映射：JSON键 -> 命令行参数名
    param_map = {
        'name': '-name',
        'output_dir': '--output_dir',
        'dataset': '--dataset',
        'dataset_dir': '--dataset_dir',
        'model': '--model',
        'seed': '--seed',
        'cuda': '--cuda',
        'num_epochs': '--num_epochs',
        'batch_size': '--batch_size',
        'shuffle': '--shuffle',
        'lr': '--lr',
        'optimizer': '--optimizer',
        'weight_decay': '--weight_decay',
        'scheduler': '--scheduler',
        'warmup_ratio': '--warmup_ratio',
        'threads': '--threads',
        'taskweights': '--taskweights',
        'featurereg': '--featurereg',
        'weightreg': '--weightreg',
        'data_aug': '--data_aug',
        'data_aug_rot': '--data_aug_rot',
        'data_aug_trans': '--data_aug_trans',
        'data_aug_scale': '--data_aug_scale',
        'label_concat': '--label_concat',
        'label_concat_reg': '--label_concat_reg',
        'labelunion': '--labelunion',
    }
    
    # 定义布尔参数的默认值（用于判断是否需要传递）
    bool_defaults = {
        'cuda': True,
        'shuffle': True,
        'taskweights': True,
        'data_aug': True,
        'featurereg': False,
        'weightreg': False,
        'label_concat': False,
        'label_concat_reg': False,
        'labelunion': False,
    }
    
    # 跳过配置文件中的 output_dir，使用传入的输出目录
    for key, value in config.items():
        if key == 'output_dir':
            continue  # 跳过配置文件中的 output_dir，使用脚本生成的路径
        
        if key in param_map:
            arg_name = param_map[key]
            
            # 处理布尔值
            if isinstance(value, bool):
                # 对于布尔参数，只在与默认值不同时才传递
                default = bool_defaults.get(key)
                if default is not None:
                    if value == default:
                        continue  # 使用默认值，跳过
                    # 如果value是False但default是True，我们无法正确传递False（argparse的type=bool限制）
                    # 这种情况下跳过参数，使用默认值True，并输出警告到stderr
                    if not value and default:
                        print(f"警告: 参数 {arg_name} 无法设置为False（默认值为True），将使用默认值True", file=sys.stderr)
                        continue
                
                args.append(arg_name)
                # 对于argparse的type=bool，传递"True"字符串会被解析为True
                if value:
                    args.append("True")
            elif value is not None:
                args.append(arg_name)
                args.append(str(value))
    
    # 添加输出目录参数（使用脚本生成的路径）
    output_dir = '$output_dir'
    if output_dir:
        args.append('--output_dir')
        args.append(output_dir)
    
    # 输出参数，每行一个参数（用于bash读取）
    # 这样可以正确处理包含空格的值
    for arg in args:
        print(arg)
    
except json.JSONDecodeError as e:
    print(f"JSON解析错误: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"错误: {e}", file=sys.stderr)
    sys.exit(1)
EOF
}

# ============================================================================
# 函数：验证配置文件
# ============================================================================
validate_config() {
    local config_file="$1"
    
    if [[ ! -f "$config_file" ]]; then
        echo -e "${RED}错误: 配置文件不存在: ${config_file}${NC}" >&2
        return 1
    fi
    
    # 使用Python验证JSON格式
    if ! python3 -m json.tool "$config_file" > /dev/null 2>&1; then
        echo -e "${RED}错误: 配置文件格式无效（不是有效的JSON）: ${config_file}${NC}" >&2
        return 1
    fi
    
    # 检查必需的参数
    local required_params=("name" "model")
    for param in "${required_params[@]}"; do
        if ! python3 -c "
import json
with open('$config_file', 'r') as f:
    cfg = json.load(f)
    if '$param' not in cfg:
        exit(1)
" 2>/dev/null; then
            echo -e "${YELLOW}警告: 配置文件中缺少参数 '${param}'，可能会使用默认值${NC}" >&2
        fi
    done
    
    return 0
}

# ============================================================================
# 函数：启动训练
# ============================================================================
start_training() {
    if [[ $# -eq 0 ]]; then
        echo -e "${RED}错误: 请指定配置文件${NC}"
        echo "使用 '$0 start <config_file>' 启动训练"
        echo "使用 '$0 configs' 查看所有可用的配置文件"
        exit 1
    fi
    
    local config_file="$1"
    
    # 如果是相对路径，尝试在CFG_DIR中查找
    if [[ ! "$config_file" =~ ^/ ]]; then
        if [[ ! -f "$config_file" ]] && [[ -f "${CFG_DIR}/${config_file}" ]]; then
            config_file="${CFG_DIR}/${config_file}"
        fi
    fi
    
    # 验证配置文件
    if ! validate_config "$config_file"; then
        exit 1
    fi
    
    # 提取任务名称和配置信息
    local job_name=$(extract_job_name_from_config "$config_file")
    local pid_file="${PID_DIR}/${job_name}.pid"
    local log_file="${LOG_DIR}/${job_name}.log"
    
    # 从配置文件读取 dataset, model, name 以构建输出目录
    local dataset=$(python3 -c "
import json
with open('$config_file', 'r') as f:
    cfg = json.load(f)
    print(cfg.get('dataset', 'unknown'))
" 2>/dev/null || echo "unknown")
    
    local model=$(python3 -c "
import json
with open('$config_file', 'r') as f:
    cfg = json.load(f)
    print(cfg.get('model', 'unknown'))
" 2>/dev/null || echo "unknown")
    
    # 构建输出目录路径: {dataset}-{model}-{name}
    local exp_output_dir="${OUTPUT_BASE_DIR}/${dataset}-${model}-${job_name}"
    
    # 创建输出目录
    mkdir -p "${exp_output_dir}"
    echo -e "${GREEN}输出目录: ${exp_output_dir}${NC}"
    
    # 检查是否已有同名任务在运行
    if [[ -f "$pid_file" ]]; then
        local old_pid=$(cat "$pid_file")
        if ps -p "$old_pid" > /dev/null 2>&1; then
            echo -e "${RED}错误: 任务 '${job_name}' 已在运行 (PID: ${old_pid})${NC}"
            exit 1
        else
            # PID文件存在但进程不存在，删除旧的PID文件
            rm -f "$pid_file"
        fi
    fi
    
    # 将配置文件转换为命令行参数（每行一个参数）
    echo -e "${GREEN}正在解析配置文件: ${config_file}${NC}"
    
    # 读取Python输出的参数到数组（传递输出目录路径）
    local train_args=()
    while IFS= read -r line; do
        [[ -n "$line" ]] && train_args+=("$line")
    done < <(config_to_args "$config_file" "$exp_output_dir")
    
    if [[ ${#train_args[@]} -eq 0 ]]; then
        echo -e "${YELLOW}警告: 配置文件解析可能有问题${NC}"
    fi
    
    # 构建用于显示的参数字符串
    local train_args_display=""
    for arg in "${train_args[@]}"; do
        if [[ "$arg" =~ [[:space:]] ]]; then
            train_args_display+=" \"$arg\""
        else
            train_args_display+=" $arg"
        fi
    done
    if [[ -n "$train_args_display" ]]; then
        train_args_display="${train_args_display:1}"  # 移除开头的空格
    fi
    
    echo -e "${GREEN}正在启动训练任务: ${job_name}${NC}"
    echo -e "配置文件: ${config_file}"
    echo -e "日志文件: ${log_file}"
    echo -e "命令: python3 ${TRAIN_SCRIPT}${train_args_display}"
    
    # 切换到scripts目录并启动训练（train_model.py期望从scripts目录运行）
    # 使用 nohup 确保进程不会因为终端关闭而中断
    # 重定向 stdout 和 stderr 到日志文件（使用绝对路径）
    local scripts_dir="$(dirname "${TRAIN_SCRIPT}")"
    # 确保日志文件路径是绝对路径
    local log_file_abs
    if [[ "$log_file" =~ ^/ ]]; then
        log_file_abs="$log_file"
    else
        log_file_abs="${SCRIPT_DIR}/${log_file}"
    fi
    
    cd "${scripts_dir}" || exit 1
    nohup python3 "$(basename "${TRAIN_SCRIPT}")" "${train_args[@]}" > "${log_file_abs}" 2>&1 &
    local pid=$!
    
    # 保存PID到文件
    echo "$pid" > "$pid_file"
    
    # 同时保存配置文件路径，方便后续查看
    echo "$config_file" > "${PID_DIR}/${job_name}.config"
    
    # 等待一下，检查进程是否成功启动
    sleep 1
    if ps -p "$pid" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ 训练任务已启动 (PID: ${pid})${NC}"
        echo -e "使用 '$0 status ${job_name}' 查看状态"
        echo -e "使用 '$0 logs ${job_name}' 查看日志"
    else
        echo -e "${RED}✗ 训练任务启动失败，请检查日志: ${log_file}${NC}"
        rm -f "$pid_file"
        rm -f "${PID_DIR}/${job_name}.config"
        exit 1
    fi
}

# ============================================================================
# 函数：停止训练
# ============================================================================
stop_training() {
    local job_name="$1"
    
    if [[ -z "$job_name" ]]; then
        # 停止所有任务
        echo -e "${YELLOW}停止所有训练任务...${NC}"
        local stopped=0
        for pid_file in "${PID_DIR}"/*.pid; do
            if [[ -f "$pid_file" ]]; then
                local name=$(basename "$pid_file" .pid)
                stop_training "$name"
                ((stopped++))
            fi
        done
        if [[ $stopped -eq 0 ]]; then
            echo -e "${YELLOW}没有正在运行的任务${NC}"
        fi
        return
    fi
    
    local pid_file="${PID_DIR}/${job_name}.pid"
    
    if [[ ! -f "$pid_file" ]]; then
        echo -e "${RED}错误: 未找到任务 '${job_name}' 的PID文件${NC}"
        return 1
    fi
    
    local pid=$(cat "$pid_file")
    
    if ! ps -p "$pid" > /dev/null 2>&1; then
        echo -e "${YELLOW}任务 '${job_name}' (PID: ${pid}) 已经停止${NC}"
        rm -f "$pid_file"
        rm -f "${PID_DIR}/${job_name}.config"
        return 0
    fi
    
    echo -e "${YELLOW}正在停止任务 '${job_name}' (PID: ${pid})...${NC}"
    
    # 发送SIGTERM信号
    kill "$pid" 2>/dev/null
    
    # 等待进程结束（最多10秒）
    local count=0
    while ps -p "$pid" > /dev/null 2>&1 && [[ $count -lt 10 ]]; do
        sleep 1
        ((count++))
    done
    
    # 如果还在运行，强制杀死
    if ps -p "$pid" > /dev/null 2>&1; then
        echo -e "${YELLOW}强制停止进程...${NC}"
        kill -9 "$pid" 2>/dev/null
        sleep 1
    fi
    
    if ps -p "$pid" > /dev/null 2>&1; then
        echo -e "${RED}✗ 无法停止任务 '${job_name}'${NC}"
        return 1
    else
        echo -e "${GREEN}✓ 任务 '${job_name}' 已停止${NC}"
        rm -f "$pid_file"
        rm -f "${PID_DIR}/${job_name}.config"
        return 0
    fi
}

# ============================================================================
# 函数：查看任务状态
# ============================================================================
show_status() {
    local job_name="$1"
    
    if [[ -z "$job_name" ]]; then
        # 显示所有任务状态
        echo -e "${GREEN}训练任务状态列表:${NC}"
        echo ""
        printf "%-30s %-10s %-15s %-20s\n" "任务名称" "状态" "PID" "日志文件"
        echo "--------------------------------------------------------------------------------"
        
        local found=0
        for pid_file in "${PID_DIR}"/*.pid; do
            if [[ -f "$pid_file" ]]; then
                found=1
                local name=$(basename "$pid_file" .pid)
                show_status "$name" | tail -1
            fi
        done
        
        if [[ $found -eq 0 ]]; then
            echo -e "${YELLOW}没有找到正在运行的任务${NC}"
        fi
        return
    fi
    
    local pid_file="${PID_DIR}/${job_name}.pid"
    local log_file="${LOG_DIR}/${job_name}.log"
    local config_file="${PID_DIR}/${job_name}.config"
    
    if [[ ! -f "$pid_file" ]]; then
        printf "%-30s %-10s %-15s %-20s\n" "$job_name" "未找到" "-" "-"
        return
    fi
    
    local pid=$(cat "$pid_file")
    local status="未知"
    local log_size=""
    
    if ps -p "$pid" > /dev/null 2>&1; then
        # 获取进程运行时间
        local runtime=$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')
        status="${GREEN}运行中${NC} (${runtime})"
    else
        status="${RED}已停止${NC}"
        rm -f "$pid_file"
        rm -f "$config_file"
    fi
    
    if [[ -f "$log_file" ]]; then
        log_size=$(du -h "$log_file" 2>/dev/null | cut -f1)
    fi
    
    local log_info="${log_file}"
    if [[ -n "$log_size" ]]; then
        log_info="${log_file} (${log_size})"
    fi
    
    printf "%-30s %b %-15s %-20s\n" "$job_name" "$status" "$pid" "$log_info"
    
    # 显示配置文件路径
    if [[ -f "$config_file" ]]; then
        echo -e "  配置文件: $(cat "$config_file")"
    fi
}

# ============================================================================
# 函数：列出所有任务
# ============================================================================
list_jobs() {
    echo -e "${GREEN}所有训练任务:${NC}"
    echo ""
    
    local found=0
    for pid_file in "${PID_DIR}"/*.pid; do
        if [[ -f "$pid_file" ]]; then
            found=1
            local name=$(basename "$pid_file" .pid)
            local config_file="${PID_DIR}/${name}.config"
            if [[ -f "$config_file" ]]; then
                echo -e "  - ${name} (配置文件: $(cat "$config_file"))"
            else
                echo -e "  - ${name}"
            fi
        fi
    done
    
    if [[ $found -eq 0 ]]; then
        echo -e "${YELLOW}没有找到任务${NC}"
    fi
}

# ============================================================================
# 函数：列出所有配置文件
# ============================================================================
list_configs() {
    echo -e "${GREEN}可用的配置文件:${NC}"
    echo ""
    
    if [[ ! -d "$CFG_DIR" ]]; then
        echo -e "${YELLOW}配置文件目录不存在: ${CFG_DIR}${NC}"
        return
    fi
    
    local found=0
    for config_file in "${CFG_DIR}"/*.json; do
        if [[ -f "$config_file" ]]; then
            found=1
            local name=$(basename "$config_file")
            local size=$(du -h "$config_file" 2>/dev/null | cut -f1)
            echo -e "  - ${name} (${size})"
            
            # 显示配置文件的name字段（如果存在）
            if command -v python3 &> /dev/null; then
                local cfg_name=$(python3 -c "
import json
try:
    with open('$config_file', 'r') as f:
        cfg = json.load(f)
        if 'name' in cfg:
            print(f\"     任务名称: {cfg['name']}\")
        if 'model' in cfg:
            print(f\"     模型: {cfg['model']}\")
        if 'num_epochs' in cfg:
            print(f\"      epochs: {cfg['num_epochs']}\")
except:
    pass
" 2>/dev/null)
                if [[ -n "$cfg_name" ]]; then
                    echo -e "$cfg_name"
                fi
            fi
        fi
    done
    
    if [[ $found -eq 0 ]]; then
        echo -e "${YELLOW}没有找到配置文件${NC}"
        echo -e "在 ${CFG_DIR} 目录下创建 .json 格式的配置文件"
    fi
}

# ============================================================================
# 函数：查看日志
# ============================================================================
view_logs() {
    local job_name="$1"
    
    if [[ -z "$job_name" ]]; then
        echo -e "${RED}错误: 请指定任务名称${NC}"
        echo "使用 '$0 list' 查看所有任务"
        exit 1
    fi
    
    local log_file="${LOG_DIR}/${job_name}.log"
    
    if [[ ! -f "$log_file" ]]; then
        echo -e "${RED}错误: 日志文件不存在: ${log_file}${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}查看日志: ${log_file}${NC}"
    echo -e "${YELLOW}按 Ctrl+C 退出${NC}"
    echo ""
    echo "--------------------------------------------------------------------------------"
    
    # 使用 tail -f 实时跟踪日志
    tail -f "$log_file"
}

# ============================================================================
# 主程序
# ============================================================================
main() {
    local command="$1"
    shift || true
    
    case "$command" in
        start)
            start_training "$@"
            ;;
        stop)
            stop_training "$@"
            ;;
        status)
            show_status "$@"
            ;;
        list)
            list_jobs
            ;;
        configs)
            list_configs
            ;;
        logs)
            view_logs "$@"
            ;;
        help|--help|-h)
            show_help
            ;;
        "")
            echo -e "${RED}错误: 请指定命令${NC}"
            echo ""
            show_help
            exit 1
            ;;
        *)
            echo -e "${RED}错误: 未知命令 '${command}'${NC}"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# 运行主程序
main "$@"
