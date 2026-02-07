#!/bin/bash
# AI Memory Plugin - Skill 安装脚本
#
# 用途：将 memory skill 安装到 Claude Code 的技能目录
#
# 使用方法：
#   bash install.sh
#
# 或直接执行：
#   chmod +x install.sh && ./install.sh

set -e  # 遇到错误时退出

# ============================================================
# 配置
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../" && pwd)"
SKILL_NAME="ai-memory"
EMBEDDING_MODEL="BAAI/bge-large-zh-v1.5"  # 默认中文模型

# ============================================================
# 颜色输出
# ============================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================================
# 检测系统
# ============================================================

detect_claude_dir() {
    # Claude Code CLI 使用 ~/.claude/
    if [[ -d "$HOME/.claude" ]]; then
        echo "$HOME/.claude"
        return 0
    fi

    # macOS Application Support
    if [[ -d "$HOME/Library/Application Support/Claude" ]]; then
        echo "$HOME/Library/Application Support/Claude"
        return 0
    fi

    # Windows
    if [[ -d "$APPDATA" ]]; then
        echo "$APPDATA\\Claude"
        return 0
    fi

    # Linux .config
    echo "$HOME/.config/Claude"
    return 0
}

# ============================================================
# 安装函数
# ============================================================

download_embedding_model() {
    log_info "检查 embedding 模型..."

    # 检查模型缓存是否存在
    CACHE_DIR="$HOME/.cache/ai-memory/models/bge-large-zh"
    if [[ -d "$CACHE_DIR" ]] && [[ -f "$CACHE_DIR/.complete" ]]; then
        log_success "Embedding 模型已缓存"
        return 0
    fi

    echo ""
    log_info "首次使用需要下载 embedding 模型（约 400MB）"
    log_info "模型: $EMBEDDING_MODEL"
    echo ""
    read -p "是否现在下载模型? (推荐) (y/n): " -n 1 -r
    echo ""

    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_warning "跳过模型下载"
        log_warning "首次使用时会自动下载，但可能需要较长时间"
        return 0
    fi

    log_info "开始下载 embedding 模型..."

    # 创建 Python 脚本下载模型
    DOWNLOAD_SCRIPT=$(cat << 'EOF'
import sys
from sentence_transformers import SentenceTransformer
from pathlib import Path

model_name = "BAAI/bge-large-zh-v1.5"
cache_dir = Path.home() / ".cache" / "ai-memory" / "models" / "bge-large-zh"

print(f"正在下载模型: {model_name}")
print(f"缓存目录: {cache_dir}")
print("")

# 下载并保存模型到缓存目录
model = SentenceTransformer(
    model_name,
    cache_folder=str(cache_dir.parent),
    trust_remote_code=True
)

# 保存模型
model.save(str(cache_dir))

# 创建完成标记
(cache_dir / ".complete").touch()

print(f"✓ 模型下载完成: {cache_dir}")
EOF
)

    # 执行下载
    if python3 -c "$DOWNLOAD_SCRIPT"; then
        log_success "Embedding 模型下载完成"
    else
        log_warning "模型下载失败，将在首次使用时自动下载"
    fi
}

install_package() {
    log_info "检查 ai-memory 包..."

    # 检查是否已安装
    if command -v ai-memory &> /dev/null; then
        log_success "ai-memory CLI 已安装"
        return 0
    fi

    log_warning "ai-memory CLI 未安装"
    echo ""
    echo "安装方式："
    echo "  1) 从 PyPI 安装（需要包已发布到 PyPI）"
    echo "  2) 从本地源码安装（开发模式）"
    echo "  3) 跳过安装（稍后手动安装）"
    read -p "请选择 (1/2/3): " -n 1 -r
    echo ""

    case "$REPLY" in
        1)
            log_info "正在从 PyPI 安装 ai-memory..."
            if pip install ai-memory; then
                log_success "ai-memory 安装成功"
            else
                log_error "ai-memory 安装失败"
                log_error "PyPI 上可能还没有此包，请尝试选项 2（本地安装）"
                exit 1
            fi
            ;;
        2)
            log_info "正在从本地源码安装（开发模式）..."
            if cd "$PROJECT_ROOT" && pip install -e .; then
                log_success "ai-memory 安装成功（开发模式）"
                log_info "注意：开发模式下，修改源码后无需重新安装"
            else
                log_error "ai-memory 安装失败"
                exit 1
            fi
            ;;
        3)
            log_info "跳过包安装"
            log_info "请稍后手动运行以下命令之一："
            log_info "  pip install ai-memory"
            log_info "  cd \"$PROJECT_ROOT\" && pip install -e ."
            return 0
            ;;
        *)
            log_error "无效选择"
            exit 1
            ;;
    esac
}

install_skill() {
    log_info "开始安装 $SKILL_NAME skill..."

    # 检测 Claude 目录
    CLAUDE_DIR=$(detect_claude_dir)
    log_info "检测到 Claude 目录: $CLAUDE_DIR"

    # 创建技能目录
    SKILL_DEST="$CLAUDE_DIR/skills/$SKILL_NAME"
    mkdir -p "$SKILL_DEST"

    # 检查 SKILL.md 文件
    SKILL_SOURCE="$SCRIPT_DIR/SKILL.md"
    if [[ ! -f "$SKILL_SOURCE" ]]; then
        log_error "未找到 skill 文件: $SKILL_SOURCE"
        exit 1
    fi

    log_info "复制 skill 文件..."
    cp "$SKILL_SOURCE" "$SKILL_DEST/SKILL.md"
    log_success "Skill 文件已复制到: $SKILL_DEST/SKILL.md"

    # 复制安装脚本（可选，方便后续更新）
    cp "$SCRIPT_DIR/install.sh" "$SKILL_DEST/install.sh"
    log_success "安装脚本已复制"

    # 检查 Claude Code 是否在运行
    if pgrep -x "Claude" > /dev/null 2>&1; then
        log_warning "检测到 Claude Code 正在运行"
        log_warning "请重启 Claude Code 以加载新 skill"
    else
        log_success "Claude Code 未运行，启动后将自动加载 skill"
    fi

    log_success "安装完成！"
    log_info "Skill 位置: $SKILL_DEST"
    log_info ""
    log_info "下一步："
    log_info "  1. 重启 Claude Code"
    log_info "  2. 在 Claude Code 中打开技能设置"
    log_info "  3. 找到 '$SKILL_NAME' skill 并启用"
}

# ============================================================
# 主流程
# ============================================================

main() {
    echo -e "${BLUE}"
    echo "============================================================"
    echo " AI Memory Plugin - Skill 安装"
    echo "============================================================"
    echo -e "${NC}"

    log_info "Skill 名称: $SKILL_NAME"

    # 安装 ai-memory 包
    install_package

    # 下载 embedding 模型
    download_embedding_model

    # 检查是否已安装 skill
    CLAUDE_DIR=$(detect_claude_dir)
    if [[ -d "$CLAUDE_DIR/skills/$SKILL_NAME" ]]; then
        log_warning "Skill 已存在于: $CLAUDE_DIR/skills/$SKILL_NAME"
        read -p "是否覆盖? (y/n): " -n 1 -r
        echo ""
        if [[ $REPLY != [yY] ]]; then
            log_info "取消安装"
            exit 0
        fi
    fi

    install_skill
}

# 运行主函数
main
