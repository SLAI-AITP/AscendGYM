"""
自定义异常类
"""


class ValidationError(Exception):
    """验证错误异常"""

    def __init__(self, message, field=None):
        """
        初始化验证错误

        参数:
            message: 错误信息
            field: 相关的字段名（可选）
        """
        self.message = message
        self.field = field
        super().__init__(self.message)

    def __str__(self):
        if self.field:
            return f"ValidationError in field '{self.field}': {self.message}"
        return f"ValidationError: {self.message}"


class ServiceError(Exception):
    """服务错误异常"""

    def __init__(self, message, service=None):
        """
        初始化服务错误

        参数:
            message: 错误信息
            service: 相关的服务名（可选）
        """
        self.message = message
        self.service = service
        super().__init__(self.message)

    def __str__(self):
        if self.service:
            return f"ServiceError in '{self.service}': {self.message}"
        return f"ServiceError: {self.message}"


class CompilationError(Exception):
    """编译错误异常"""

    def __init__(self, message, errors=None):
        """
        初始化编译错误

        参数:
            message: 错误信息
            errors: 详细的编译错误列表（可选）
        """
        self.message = message
        self.errors = errors or []
        super().__init__(self.message)

    def __str__(self):
        error_count = len(self.errors)
        if error_count > 0:
            return f"CompilationError: {self.message} ({error_count} errors)"
        return f"CompilationError: {self.message}"


class TestError(Exception):
    """测试错误异常"""

    def __init__(self, message, failed_cases=None):
        """
        初始化测试错误

        参数:
            message: 错误信息
            failed_cases: 失败的测试用例列表（可选）
        """
        self.message = message
        self.failed_cases = failed_cases or []
        super().__init__(self.message)

    def __str__(self):
        failed_count = len(self.failed_cases)
        if failed_count > 0:
            return f"TestError: {self.message} ({failed_count} failed cases)"
        return f"TestError: {self.message}"


class ConfigurationError(Exception):
    """配置错误异常"""

    def __init__(self, message, config_key=None):
        """
        初始化配置错误

        参数:
            message: 错误信息
            config_key: 相关的配置键（可选）
        """
        self.message = message
        self.config_key = config_key
        super().__init__(self.message)

    def __str__(self):
        if self.config_key:
            return f"ConfigurationError for '{self.config_key}': {self.message}"
        return f"ConfigurationError: {self.message}"