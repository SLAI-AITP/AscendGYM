"""
RAG问答服务
功能：并发调用问答接口，获取大模型总结答案
"""
import logging
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from config.settings import WINDVECTOR_CONFIG

logger = logging.getLogger(__name__)


class WindRAGSearchService:
    """WindRAG问答服务（支持并发多问题问答）"""

    def __init__(self):
        """初始化服务"""
        self.qa_url = WINDVECTOR_CONFIG['rag_qa_url']
        self.original_num = int(WINDVECTOR_CONFIG['windvectord_original_num'])
        self.default_timeout = 600  # 问答接口超时时间（秒）
        logger.info("WindRAGQAService initialized")

    def search(self, queries: List[str], top_k: int = 5) -> Dict[str, Any]:
        """
        并发问答多个问题

        参数:
            queries: 问题列表，1-5个问题
            top_k: 每个问题rerank后返回的文档数量，默认5，范围1-10
            history_messages: 可选的历史对话消息
            stream: 是否流式返回
            enable_thinking: 是否启用思考模式

        返回:
            dict: 包含问答结果的字典
        """
        logger.info(f"Starting QA for {len(queries)} queries with top_k={top_k}")

        try:
            # 并发调用问答接口
            results = self._concurrent_search(queries, top_k)

            # 统计成功/失败数量
            success_count = sum(1 for r in results if r.get('success', False))

            # 组织最终结果
            final_result = {
                'success': True,
                'message': f'成功回答 {success_count}/{len(queries)} 个问题',
                'query_count': len(queries),
                'success_count': success_count,
                'top_k': top_k,
                'results': results
            }

            logger.info(f"Search completed: {success_count}/{len(queries)} queries successful")
            return final_result

        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return {
                'success': False,
                'message': f'问答过程出现异常: {str(e)}',
                'query_count': len(queries),
                'results': []
            }

    def _concurrent_search(self, queries: List[str], top_k: int) -> List[Dict[str, Any]]:
        """
        并发调用问答接口
        使用线程池实现并发，保持结果顺序
        """
        query_to_index = {query: i for i, query in enumerate(queries)}
        result_map = {}

        # 使用线程池并发问答
        max_workers = min(len(queries), 5)
        logger.info(f"Using {max_workers} workers for concurrent QA")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有问答任务
            future_to_query = {
                executor.submit(self._search_single_query, query, top_k): query for query in queries
            }

            # 收集结果
            for future in as_completed(future_to_query):
                query = future_to_query[future]
                index = query_to_index[query]
                try:
                    result = future.result(timeout=self.default_timeout + 10)
                    result_map[index] = result
                    logger.info(f"Query '{query[:50]}...' answered successfully")
                except Exception as e:
                    logger.error(f"Failed to get answer for query '{query[:50]}...': {str(e)}")
                    result_map[index] = {
                        'query': query,
                        'success': False,
                        'error': str(e),
                        'answer': None
                    }

        # 按原始顺序返回结果
        results = [result_map[i] for i in range(len(queries))]
        return results

    def _search_single_query(self, query: str, top_k: int) -> Dict[str, Any]:
        """
        调用问答接口回答单个问题

        返回格式:
        {
            'query': 原始问题,
            'success': 是否成功,
            'answer': 大模型总结的答案,
            'search_info': 检索信息摘要（可选）
        }
        """
        logger.info(f"Asking query: {query[:100]}")

        try:
            # 构建请求体
            request_body = {
                "query": query,
                "history_messages": [],
                "windvector_config": {
                    "top_k": self.original_num,
                    "rerank_top_k": top_k
                },
                "stream": False,
                "enable_thinking": False,
                "mode": "auto",
                "developer_mode": True
            }

            headers = {"Content-Type": "application/json"}

            # 调用问答接口
            response = requests.post(
                self.qa_url,
                headers=headers,
                json=request_body,
                verify=False,
                timeout=self.default_timeout
            )

            logger.info(f"QA URL: {self.qa_url}")

            if response.status_code != 200:
                error_msg = f"HTTP {response.status_code}"
                logger.error(f"Search failed for query: {error_msg}")
                return {
                    'query': query,
                    'success': False,
                    'error': f'Search API error: {error_msg}',
                    'answer': None
                }

            response_data = response.json()

            # 提取答案
            answer = self._extract_answer(response_data)
            # 提取模式
            mode = response_data.get("search_content", {}).get("mode", "")

            if answer is None:
                logger.warning(f"No answer extracted for query '{query[:50]}...'")
                return {
                    'query': query,
                    'success': False,
                    'error': 'Failed to extract answer from response',
                    'answer': None
                }

            result = {
                'query': query,
                'success': True,
                'answer': answer,
                'mode': mode
            }

            logger.info(f"Successfully answered query '{query[:50]}...'")
            return result

        except requests.exceptions.Timeout:
            logger.error(f"Timeout answering query '{query[:50]}...'")
            return {
                'query': query,
                'success': False,
                'error': 'Request timeout',
                'answer': None
            }
        except Exception as e:
            logger.error(f"Error answering query '{query[:50]}...': {str(e)}")
            return {
                'query': query,
                'success': False,
                'error': str(e),
                'answer': None
            }

    def _extract_answer(self, response_data: Dict) -> Optional[str]:
        """
        从响应中提取答案

        响应格式:
        {
            "choices": [{
                "message": {
                    "content": "答案内容"
                }
            }]
        }
        """
        try:
            choices = response_data.get("choices", [])
            if not choices:
                return None

            message = choices[0].get("message", {})
            content = message.get("content")

            return content if content else None

        except Exception as e:
            logger.error(f"Error extracting answer: {str(e)}")
            return None
