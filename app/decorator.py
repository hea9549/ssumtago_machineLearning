from functools import wraps
from flask import Flask, session, redirect, url_for, escape, request, jsonify
from jsonschema import validate, ValidationError

def json_schema(schema_name,app):
    """
    지정한 API 에 대해서 지정한 schema_name로 검사한다.
    :param schema_name: 검사대상 스키마 이름
    :return: 에러나면 40000 에러
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kw):
            try:
                # request.on_json_loading_failed = on_json_loading_failed_return_dict
                validate(request.json, app.config["schema"][schema_name])
                return func(*args, **kw)
            except ValidationError as e:
                print(e)
                return "invalid json format"
                # logger.exception(traceback.format_exc())
                # return ResponseData(code=HttpStatusCode.INVALID_PARAMETER).json
        return wrapper

    return decorator

