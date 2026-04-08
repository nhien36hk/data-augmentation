import re

import numpy as np
from tree_sitter import Node

from src.data_preprocessors.language_processors.utils import get_tokens, dfs_print


class JavaAndCPPProcessor:
    @classmethod
    def create_dead_for_loop(cls, body):
        control_variable = "_i_" + str(np.random.choice(list(range(10))))
        p = np.random.uniform(0, 1)
        if p < 0.5:
            # Defines the variable, so this is valid C99+/Java/C++
            prefix = "for ( int " + control_variable + " = 0 ; " + control_variable + " > 0 ; " + control_variable + \
                     " ++ ) { "
            loop = prefix + body + " } "
            return loop
        else:
            return "for ( ; false ; ) { " + body + "}"

    @classmethod
    def create_dead_while_loop(cls, body):
        p = np.random.uniform(0, 1)
        # Use constants to avoid "undeclared identifier" errors
        if p < 0.33:
            return "while ( false ) { " + body + " }"
        elif p < 0.66:
            return "while ( 0 < 0 ) { " + body + " } "
        else:
            return "while ( 0 > 1 ) { " + body + " } "

    @classmethod
    def create_dead_if(cls, body):
        p = np.random.uniform(0, 1)
        # Use constants to avoid "undeclared identifier" errors
        if p < 0.33:
            return "if ( false ) { " + body + " }"
        elif p < 0.66:
            return "if ( 0 < 0 ) { " + body + " } "
        else:
            return "if ( 0 > 1 ) { " + body + " } "

    @classmethod
    def for_to_while_random(cls, code_string, parser):
        root = parser.parse_code(code_string)
        loops = cls.extract_for_loops(root)
        success = False
        try:
            while not success and len(loops) > 0:
                selected_loop = np.random.choice(loops)
                loops.remove(selected_loop)
                modified_root, modified_code_string, success = JavaAndCPPProcessor.for_to_while(
                    code_string, root, selected_loop, parser
                )
                if success:
                    root = modified_root
                    code_string = modified_code_string
        except:
            pass
        if not success:
            code_string = cls.beautify_java_code(get_tokens(code_string, root))
        return root, code_string, success

    @classmethod
    def while_to_for_random(cls, code_string, parser):
        root = parser.parse_code(code_string)
        loops = cls.extract_while_loops(root)
        success = False
        try:
            while not success and len(loops) > 0:
                selected_loop = np.random.choice(loops)
                loops.remove(selected_loop)
                modified_root, modified_code_string, success = JavaAndCPPProcessor.while_to_for(
                    code_string, root, selected_loop, parser
                )
                if success:
                    root = modified_root
                    code_string = modified_code_string
            if not success:
                code_string = cls.beautify_java_code(get_tokens(code_string, root))
        except:
            pass
        return root, code_string, success

    @classmethod
    def extract_for_loops(cls, root):
        loops = []
        queue = [root]
        while len(queue) > 0:
            current_node = queue[0]
            queue = queue[1:]
            if str(current_node.type) == 'for_statement':
                loops.append(current_node)
            for child in current_node.children:
                queue.append(child)
        return loops

    @classmethod
    def beautify_java_code(cls, tokens):
        # Heuristic beautification to avoid single-line output
        code = ""
        indent_level = 0
        indent_str = "    "
        paren_depth = 0  # Track parenthesis depth to avoid breaking lines inside (for loops)
        
        # Token-based reconstruction with newline logic
        for i, token in enumerate(tokens):
            if token == "{":
                code += " {\n"
                indent_level += 1
                code += indent_str * indent_level
            elif token == "}":
                indent_level = max(0, indent_level - 1)
                code += "\n" + (indent_str * indent_level) + "}"
                # Add newline after closing brace unless followed by another closing brace or else/catch
                if i + 1 < len(tokens) and tokens[i+1] not in ["}", "else", "catch", ";"]:
                    code += "\n" + (indent_str * indent_level)
            elif token == "(":
                code += " ("
                paren_depth += 1
            elif token == ")":
                code += ")"
                paren_depth = max(0, paren_depth - 1)
            elif token == ";":
                code += ";"
                # Only newline if NOT inside parentheses (e.g. for loop header)
                if paren_depth == 0:
                    code += "\n" + (indent_str * indent_level)
            else:
                # Add space before token if not at start of line and previous token wasn't open paren/bracket logic (simplified)
                if len(code) > 0 and code[-1] not in [" ", "\n", "("]:
                     code += " "
                code += token

        # Post-process cleanup
        code = re.sub(r"\s*\.\s*", ".", code)      # Fix "obj . method" or "obj. method" -> "obj.method"
        code = re.sub(r"\s+\+\+", "++", code)      # Fix "i ++" -> "i++"
        code = re.sub(r"\(\s+", "(", code)         # Fix "( expr" -> "(expr"
        code = re.sub(r"\s+\)", ")", code)         # Fix "expr )" -> "expr)"
        code = re.sub(r"\s+;", ";", code)          # Fix " ;" -> ";"
        code = re.sub(r"\s*\[\s*", "[", code)      # Fix "arr [ i ]" -> "arr[i"
        code = re.sub(r"\s*\]", "]", code)         # Fix "i ]" -> "i]"
        code = re.sub(r"\s*,\s*", ", ", code)      # Fix "a , b" -> "a, b"
        # Fix space after open paren created by heuristic
        code = re.sub(r" \(", "(", code)
        # Ensure space before open brace
        code = re.sub(r"([^\s])\{", r"\1 {", code)
        
        return code.strip()

    @classmethod
    def get_tokens_replace_for(cls, code_str, for_node, root, init, cond, update, body):
        if isinstance(code_str, str):
            code_str = code_str.encode()
        assert isinstance(root, Node)
        tokens = []
        if "comment" in str(root.type):
            comment_content = code_str[root.start_byte:root.end_byte].decode().strip()
            tokens.append(comment_content + "\n")
            return tokens
        if "string" in str(root.type):
            return [code_str[root.start_byte:root.end_byte].decode()]
        children = root.children
        if len(children) == 0:
            tokens.append(code_str[root.start_byte:root.end_byte].decode())
        for child in children:
            if child == for_node:
                tokens.extend(
                    init + ["while", "("] + cond + [")", "{"] + body + update + ["}"]
                )
            else:
                tokens += JavaAndCPPProcessor.get_tokens_replace_for(code_str, for_node, child, init, cond, update,
                                                                     body)
        return tokens

    @classmethod
    def extract_for_contents(cls, for_loop, code_string):
        children = for_loop.children
        init_part = children[2]
        if str(init_part.type).endswith("expression"):
            next_part_start = 4
            init_tokens = get_tokens(code_string, init_part) + [";"]
        elif str(init_part.type).endswith("statement") or str(init_part.type).endswith("declaration"):
            next_part_start = 3
            init_tokens = get_tokens(code_string, init_part)
        else:
            next_part_start = 3
            init_tokens = []
        comp_part = children[next_part_start]
        if str(comp_part.type).endswith("expression"):
            next_part_start += 2
            comp_tokens = get_tokens(code_string, comp_part)
        else:
            comp_tokens = ["true"]
            next_part_start += 1
        update_part = children[next_part_start]
        if str(update_part.type).endswith("expression"):
            next_part_start += 2
            update_tokens = get_tokens(code_string, update_part) + [";"]
        else:
            update_tokens = []
            next_part_start += 1
        block_part = children[next_part_start]
        breaking_statements = cls.get_breaking_statements(block_part)
        block_tokens = cls.get_tokens_insert_before(
            code_string, block_part, " ".join(update_tokens), breaking_statements)
        return init_tokens, comp_tokens, update_tokens, block_tokens

    @classmethod
    def get_tokens_insert_before(cls, code_str, root, insertion_code, insert_before_node):
        if not isinstance(insert_before_node, list):
            insert_before_node = [insert_before_node]
        if isinstance(code_str, str):
            code_str = code_str.encode()
        assert isinstance(root, Node)
        tokens = []
        
        # Handling comments: ensure they are followed by newline to not consume next code
        if "comment" in str(root.type): # Matches "comment", "line_comment", "block_comment"
            comment_content = code_str[root.start_byte:root.end_byte].decode().strip()
            tokens.append(comment_content + "\n")
            return tokens
            
        if "string" in str(root.type):
            parent = root.parent
            if len(parent.children) == 1:
                return tokens
            else:
                return [code_str[root.start_byte:root.end_byte].decode()]
        
        # INSERTION LOGIC ----------------------------------------------------
        # Special case: If we are inserting before a node, we must check if that node
        # is a direct child of a control statement (if, while, etc.) but NOT a block.
        # If so, we must upgrade the single statement to a block { ... } to hold both
        # the insertion code (update) and the original statement (break/continue).
        
        should_insert = root in insert_before_node
        
        # Check if we need to wrap in braces
        # We need wrapping if we are inserting AND the parent is an IF/ELSE/WHILE/FOR
        # but the parent is NOT a compound_statement/block.
        needs_block_wrap = False
        if should_insert and root.parent:
            ptype = str(root.parent.type)
            if ptype in ["if_statement", "while_statement", "for_statement", "else_clause"] or ptype.endswith("_statement"):
                # If the parent expects a statement and we are replacing a single statement
                # with multiple statments (insertion + original), we must wrap.
                # However, usually the parser sees the block as a child.
                # If 'root' is NOT a block/compound_statement, we wrap.
                if str(root.type) not in ["compound_statement", "block"]:
                    needs_block_wrap = True

        if should_insert:
            if needs_block_wrap:
                tokens.append("{")
            
            tokens += insertion_code.split()
            
            # If we wrap, the insertion code is inside the block.
            # Then we process the root (the break/continue statement).
        
        children = root.children
        if len(children) == 0:
            # Leaf node
            tokens.append(code_str[root.start_byte:root.end_byte].decode())
        else:
            # Recursive processing
            for child in children:
                ts = cls.get_tokens_insert_before(code_str, child, insertion_code, insert_before_node)
                tokens += ts
        
        if should_insert and needs_block_wrap:
            tokens.append("}")
            
        return tokens

    @classmethod
    def get_breaking_statements(cls, block):
        # We only care about 'continue' statements for inserting loop updates (e.g. i++).
        # 'break' and 'return' exit the loop immediately without running the update step in a for-loop,
        # so we should NOT insert updates before them.
        breakings = ['continue_statement']
        
        # Stop traversing into nested loops, because their 'continue'/'break' belong to them (unless labeled, but simple heuristic first)
        loop_types = ['for_statement', 'while_statement', 'do_statement']
        
        statements = []
        stack = [block]
        while len(stack) > 0:
            top = stack.pop()
            if str(top.type) in breakings:
                statements.append(top)
            
            # Use 'children' for traversal
            # But do NOT traverse into nested loops
            if str(top.type) not in loop_types:
                for child in top.children:
                    stack.append(child)
        return statements

    @classmethod
    def for_to_while(cls, code_string, root, fl, parser):
        original_tokenized_code = " ".join(get_tokens(code_string, root))
        init_tokens, comp_tokens, update_tokens, body_tokens = cls.extract_for_contents(fl, code_string)
        if len(body_tokens) >= 2 and (body_tokens[0] == "{" and body_tokens[-1] == "}"):
            body_tokens = body_tokens[1:-1]
        tokens = cls.get_tokens_replace_for(
            code_str=code_string,
            for_node=fl,
            root=root,
            init=init_tokens,
            cond=comp_tokens,
            update=update_tokens,
            body=body_tokens
        )
        if original_tokenized_code == " ".join(tokens):
            return root, original_tokenized_code, False
        code = cls.beautify_java_code(tokens)
        return parser.parse_code(code), code, True

    @classmethod
    def extract_while_loops(cls, root):
        loops = []
        queue = [root]
        while len(queue) > 0:
            current_node = queue[0]
            queue = queue[1:]
            if str(current_node.type) == 'while_statement':
                loops.append(current_node)
            for child in current_node.children:
                queue.append(child)
        return loops

    @classmethod
    def while_to_for(cls, code_string, root, wl, parser):
        children = wl.children
        condition = children[1]
        body = children[2]
        if str(condition.type) == 'parenthesized_expression':
            expr_tokens = get_tokens(code_string, condition.children[1])
            body_tokens = get_tokens(code_string, body)
            if len(body_tokens) >= 2 and (body_tokens[0] == "{" and body_tokens[-1] == "}"):
                body_tokens = body_tokens[1:-1]
            tokens = cls.get_tokens_replace_while(
                code_str=code_string,
                while_node=wl,
                root=root,
                cond=expr_tokens,
                body=body_tokens
            )
            code = cls.beautify_java_code(tokens)
            return parser.parse_code(code), code, True
        return root, code_string, False

    @classmethod
    def get_tokens_replace_while(cls, code_str, while_node, root, cond, body):
        if isinstance(code_str, str):
            code_str = code_str.encode()
        assert isinstance(root, Node)
        tokens = []
        if "comment" in str(root.type):
            comment_content = code_str[root.start_byte:root.end_byte].decode().strip()
            tokens.append(comment_content + "\n")
            return tokens
        if "string" in str(root.type):
            return [code_str[root.start_byte:root.end_byte].decode()]
        children = root.children
        if len(children) == 0:
            tokens.append(code_str[root.start_byte:root.end_byte].decode())
        for child in children:
            if child == while_node:
                tokens.extend(
                    ["for", "(", ";"] + cond + [";", ")", "{"] + body + ["}"]
                )
            else:
                tokens += JavaAndCPPProcessor.get_tokens_replace_while(code_str, while_node, child, cond, body)
        return tokens

    # -----Confusion removal C------
    # -----Confusion removal C------
    @classmethod
    def conditional_removal(cls, code_string, parser):
        # This function is for C, equavalent to extract_ternary_expression for Java
        root = parser.parse_code(code_string)
        assi_con_expr, varde_con_expr, ret_con_expr = cls.extract_conditional_expression(root)
        
        replacements = []
        if isinstance(code_string, str):
            code_bytes = code_string.encode('utf-8')
        else:
            code_bytes = code_string
            
        # 1. Assignment Conditional
        for node in assi_con_expr:
            try:
                # node could be expression_statement or assignment_expression
                # Find the assignment_expression that contains the conditional
                assign_expr = None
                if str(node.type) == 'expression_statement':
                    # Children: [assignment_expression, ;]
                    for child in node.children:
                        if str(child.type) == 'assignment_expression':
                            assign_expr = child
                            break
                elif str(node.type) == 'assignment_expression':
                    assign_expr = node
                
                if not assign_expr or len(assign_expr.children) < 3:
                    continue
                
                assignee = assign_expr.children[0]
                right = assign_expr.children[2]
                
                if str(right.type) != 'conditional_expression':
                    continue
                
                cond_expr = right
                cond = cond_expr.child_by_field_name('condition')
                conseq = cond_expr.child_by_field_name('consequence')
                alt = cond_expr.child_by_field_name('alternative')
                
                if not (cond and conseq and alt):
                     cond = cond_expr.children[0]
                     conseq = cond_expr.children[2]
                     alt = cond_expr.children[4]

                assignee_text = code_bytes[assignee.start_byte:assignee.end_byte].decode()
                cond_text = code_bytes[cond.start_byte:cond.end_byte].decode()
                if cond.type != 'parenthesized_expression': cond_text = f"({cond_text})"
                
                true_text = code_bytes[conseq.start_byte:conseq.end_byte].decode()
                false_text = code_bytes[alt.start_byte:alt.end_byte].decode()
                
                new_text = f"if {cond_text} {{ {assignee_text} = {true_text}; }} else {{ {assignee_text} = {false_text}; }}"
                replacements.append((node.start_byte, node.end_byte, new_text))
            except:
                pass
                
        # 2. Variable Declaration (init_declarator)
        for node in varde_con_expr:
             # node is declaration (parent of init_declarator)
             
             # Check for multiple declarators
             declarators = [c for c in node.children if c.type == 'init_declarator']
             if len(declarators) > 1: continue

             for child in node.children:
                 if child.type == 'init_declarator':
                     try:
                         # init_declarator: [declarator, =, value]
                         if len(child.children) < 3: continue
                         val = child.children[2]
                         if val.type != 'conditional_expression': continue
                         
                         assignee = child.children[0]
                         cond_expr = val
                         
                         cond = cond_expr.child_by_field_name('condition')
                         conseq = cond_expr.child_by_field_name('consequence')
                         alt = cond_expr.child_by_field_name('alternative')
                         
                         if not (cond and conseq and alt):
                             cond = cond_expr.children[0]
                             conseq = cond_expr.children[2]
                             alt = cond_expr.children[4]
                             
                         # Preserve modifiers/type using prefix
                         prefix = code_bytes[node.start_byte:child.start_byte].decode()
                         assignee_text = code_bytes[assignee.start_byte:assignee.end_byte].decode()
                         
                         cond_text = code_bytes[cond.start_byte:cond.end_byte].decode()
                         if cond.type != 'parenthesized_expression': cond_text = f"({cond_text})"
                         
                         true_text = code_bytes[conseq.start_byte:conseq.end_byte].decode()
                         false_text = code_bytes[alt.start_byte:alt.end_byte].decode()
                         
                         new_text = f"{prefix}{assignee_text}; if {cond_text} {{ {assignee_text} = {true_text}; }} else {{ {assignee_text} = {false_text}; }}"
                         replacements.append((node.start_byte, node.end_byte, new_text))

                     except:
                         pass

        # 3. Return Statement
        for node in ret_con_expr:
            try:
                # return expr;
                # children: [return, expr, ;]
                if len(node.children) < 2: continue
                expr = node.children[1]
                if expr.type != 'conditional_expression': continue
                
                cond_expr = expr
                cond = cond_expr.child_by_field_name('condition')
                conseq = cond_expr.child_by_field_name('consequence')
                alt = cond_expr.child_by_field_name('alternative')
                
                if not (cond and conseq and alt):
                     cond = cond_expr.children[0]
                     conseq = cond_expr.children[2]
                     alt = cond_expr.children[4]
                     
                cond_text = code_bytes[cond.start_byte:cond.end_byte].decode()
                if cond.type != 'parenthesized_expression': cond_text = f"({cond_text})"
                
                true_text = code_bytes[conseq.start_byte:conseq.end_byte].decode()
                false_text = code_bytes[alt.start_byte:alt.end_byte].decode()
                
                new_text = f"if {cond_text} {{ return {true_text}; }} else {{ return {false_text}; }}"
                replacements.append((node.start_byte, node.end_byte, new_text))
            except:
                pass
                
        if replacements:
            new_code = cls.apply_replacements(code_string, replacements)
            return parser.parse_code(new_code), new_code, True

        return root, code_string, False

    @classmethod
    def assignment_conditional_removal(cls, code_string, assi_tern_expr, root, parser):
        if isinstance(code_string, str):
            code_string = code_string.encode()
        assert isinstance(root, Node)
        tokens = []
        children = root.children
        if len(children) == 0:
            tokens.append(cls.handle_terminal_node(root, code_string))
        for child in children:
            if child in assi_tern_expr:
                if str(child.children[0].type) == "conditional_expression":
                    cond_children = child.children[0].children
                    if str(cond_children[0].type) == "assignment_expression":
                        assignee_token = get_tokens(code_string, cond_children[0].children[0])[0]
                        condition_tokens = get_tokens(code_string, cond_children[0].children[2])
                        if str(cond_children[0].children[2].type) == 'parenthesized_expression':
                            condition_tokens = condition_tokens[1:-1]
                        br1_tokens = get_tokens(code_string, cond_children[2])
                        br2_tokens = get_tokens(code_string, cond_children[4])
                        tokens.extend(["if", "("] + condition_tokens + [")", "{", assignee_token, "="] + br1_tokens +
                                      [";", "}", "else", "{", assignee_token, "="] + br2_tokens + [";", "}"])
            else:
                tokens += JavaAndCPPProcessor.assignment_conditional_removal(code_string, assi_tern_expr, child, parser)
        return tokens

    @classmethod
    def extract_conditional_expression(cls, root):
        assi_con_expr = []
        varde_con_expr = []
        ret_con_expr = []
        queue = [root]
        while len(queue) > 0:
            current_node = queue[0]
            queue = queue[1:]
            
            # Check for conditional_expression (ternary operator)
            if str(current_node.type) == 'conditional_expression':
                parent = current_node.parent
                parent_type = str(parent.type) if parent else ""
                
                # Case 1: Assignment expression: a = cond ? x : y;
                # Parent is assignment_expression, and conditional is on the right side
                if parent_type == "assignment_expression":
                    # Get the grandparent (expression_statement) for replacement
                    grandparent = parent.parent
                    if grandparent and str(grandparent.type) == "expression_statement":
                        assi_con_expr.append(grandparent)
                    else:
                        assi_con_expr.append(parent)
                
                # Case 2: Variable declaration: int x = cond ? a : b;
                elif parent_type == "init_declarator":
                    varde_con_expr.append(parent.parent)  # node type: declaration
                
                # Case 3: Return statement: return cond ? a : b;
                elif parent_type == "return_statement":
                    ret_con_expr.append(parent)
            
            for child in current_node.children:
                queue.append(child)
        return assi_con_expr, varde_con_expr, ret_con_expr

    # -----Confusion removal Java------
    # TODO: Check whether java/C/CPP have the same "ternary_expression" node type
    @classmethod
    def apply_replacements(cls, code_string, replacements):
        if isinstance(code_string, str):
            code_bytes = bytearray(code_string, 'utf-8')
        else:
            code_bytes = bytearray(code_string)
        
        # Sort replacements by start_byte descending to avoid offset issues
        # replacements is list of (start, end, new_bytes)
        replacements.sort(key=lambda x: x[0], reverse=True)

        for start, end, new_text in replacements:
            if isinstance(new_text, str):
                new_text = new_text.encode('utf-8')
            code_bytes[start:end] = new_text
            
        return code_bytes.decode('utf-8')

    # -----Confusion removal Java------
    @classmethod
    def ternary_removal(cls, code_string, parser):
        # code_string = cls.remove_package_and_import(code_string) # Avoid removing imports for now
        root = parser.parse_code(code_string)
        assi_tern_expr, varde_tern_expr, ret_tern_expr = cls.extract_ternary_expression(root)
        
        replacements = []
        
        if isinstance(code_string, str):
            code_bytes = code_string.encode('utf-8')
        else:
            code_bytes = code_string
            
        # 1. Assignment Ternary
        for node in assi_tern_expr:
            # node is assignment_expression
            # children: [assignee, =, ternary_expr]
            # ternary_expr children: [cond, ?, true_val, :, false_val]
            try:
                children = node.children
                if len(children) < 3: continue
                assignee = children[0]
                ternary = children[2]
                if ternary.type != 'ternary_expression': continue
                
                cond = ternary.child_by_field_name('condition')
                conseq = ternary.child_by_field_name('consequence')
                alt = ternary.child_by_field_name('alternative')
                
                if not (cond and conseq and alt):
                     # Fallback to index if fields missing
                     cond = ternary.children[0]
                     conseq = ternary.children[2]
                     alt = ternary.children[4]
                
                assignee_text = code_bytes[assignee.start_byte:assignee.end_byte].decode()
                cond_text = code_bytes[cond.start_byte:cond.end_byte].decode()
                
                # Check formatting of condition
                if cond.type == 'parenthesized_expression':
                     # keep parens
                     pass
                else:
                     cond_text = f"({cond_text})"
                     
                true_text = code_bytes[conseq.start_byte:conseq.end_byte].decode()
                false_text = code_bytes[alt.start_byte:alt.end_byte].decode()
                
                new_text = f"if {cond_text} {{ {assignee_text} = {true_text}; }} else {{ {assignee_text} = {false_text}; }}"
                
                replacements.append((node.start_byte, node.end_byte, new_text))
            except:
                pass

        # 2. Variable Declaration Ternary
        for node in varde_tern_expr:
            # node is local_variable_declaration (or similar parent) containing the declarator
            
            # Check for multiple declarators to avoid side effects
            declarators = [c for c in node.children if c.type in ["variable_declarator", "init_declarator"]]
            if len(declarators) > 1: continue

            # We iterate children to find the specific declarator with ternary
            for child in node.children:
                if child.type in ["variable_declarator", "init_declarator"]:
                    # Child: [name, =, ternary]
                    try:
                        decl_children = child.children
                        if len(decl_children) < 3: continue
                        if decl_children[-1].type != 'ternary_expression': continue
                        
                        assignee = decl_children[0]
                        ternary = decl_children[-1]
                        
                        cond = ternary.child_by_field_name('condition')
                        conseq = ternary.child_by_field_name('consequence') 
                        alt = ternary.child_by_field_name('alternative')
                        
                        if not (cond and conseq and alt):
                             cond = ternary.children[0]
                             conseq = ternary.children[2]
                             alt = ternary.children[4]

                        assignee_text = code_bytes[assignee.start_byte:assignee.end_byte].decode() 
                        
                        # Preserve type + modifiers by taking everything before the declarator
                        prefix = code_bytes[node.start_byte:child.start_byte].decode()
                        # prefix usually ends with space. Clean if needed but usually fine.
                        
                        cond_text = code_bytes[cond.start_byte:cond.end_byte].decode()
                        if cond.type != 'parenthesized_expression': cond_text = f"({cond_text})"
                        
                        true_text = code_bytes[conseq.start_byte:conseq.end_byte].decode()
                        false_text = code_bytes[alt.start_byte:alt.end_byte].decode()
                        
                        new_text = f"{prefix}{assignee_text}; if {cond_text} {{ {assignee_text} = {true_text}; }} else {{ {assignee_text} = {false_text}; }}"
                        
                        replacements.append((node.start_byte, node.end_byte, new_text))
                    except:
                        pass

        # 3. Return Ternary
        for node in ret_tern_expr:
            # node is return_statement
            try:
                ternary = node.children[1] # return EXPR ;
                if ternary.type != 'ternary_expression': continue
                
                cond = ternary.child_by_field_name('condition')
                conseq = ternary.child_by_field_name('consequence')
                alt = ternary.child_by_field_name('alternative')
                
                if not (cond and conseq and alt):
                     cond = ternary.children[0]
                     conseq = ternary.children[2]
                     alt = ternary.children[4]
                     
                cond_text = code_bytes[cond.start_byte:cond.end_byte].decode()
                if cond.type != 'parenthesized_expression': cond_text = f"({cond_text})"
                
                true_text = code_bytes[conseq.start_byte:conseq.end_byte].decode()
                false_text = code_bytes[alt.start_byte:alt.end_byte].decode()
                
                new_text = f"if {cond_text} {{ return {true_text}; }} else {{ return {false_text}; }}"
                
                replacements.append((node.start_byte, node.end_byte, new_text))
            except:
                pass
        
        if replacements:
            new_code = cls.apply_replacements(code_string, replacements)
            return parser.parse_code(new_code), new_code, True
            
        return root, code_string, False

    @classmethod
    def assignment_ternary_removal(cls, code_string, assi_tern_expr, root, parser):
        if isinstance(code_string, str):
            code_string = code_string.encode()
        assert isinstance(root, Node)
        tokens = []
        children = root.children
        if len(children) == 0:
            tokens.append(cls.handle_terminal_node(root, code_string))
        for child in children:
            if child in assi_tern_expr:
                te_children = child.children
                assignee = te_children[0]
                # children[1] should be "="
                body = te_children[2]
                tokens = cls.ternary_body_write(body, code_string, assignee, tokens)
                break
            else:
                tokens += JavaAndCPPProcessor.assignment_ternary_removal(code_string, assi_tern_expr, child, parser)
        return tokens

    @classmethod
    def var_decl_ternary_removal(cls, code_string, var_decl_tern_expr, root, parser):
        if isinstance(code_string, str):
            code_string = code_string.encode()
        assert isinstance(root, Node)
        tokens = []
        children = root.children
        if len(children) == 0:
            tokens.append(cls.handle_terminal_node(root, code_string))
        for child in children:
            if child in var_decl_tern_expr:
                # tokens.extend(["return"])
                for c in child.children:
                    if str(c.type) == ";":
                        continue
                    elif str(c.type) == "variable_declarator" or str(c.type) == "init_declarator":  # the former is
                        # for java and the latter is for c:
                        te_children = c.children
                        assignee = te_children[0]
                        assignee_token = get_tokens(code_string, assignee)[0]
                        tokens.extend([assignee_token, ";"])
                        # children[1] should be "="
                        body = te_children[2]
                        tokens = cls.ternary_body_write(body, code_string, assignee, tokens)
                    else:
                        tokens += get_tokens(code_string, c)
            else:
                tokens += JavaAndCPPProcessor.var_decl_ternary_removal(code_string, var_decl_tern_expr, child, parser)
        return tokens

    @classmethod
    def return_ternary_removal(cls, code_string, ret_tern_expr, root, parser):
        if isinstance(code_string, str):
            code_string = code_string.encode()
        assert isinstance(root, Node)
        tokens = []
        children = root.children
        if len(children) == 0:
            tokens.append(cls.handle_terminal_node(root, code_string))
        for child in children:
            if child in ret_tern_expr:
                te_children = child.children
                assignee = te_children[0]
                # children[1] should be "="
                body = te_children[1]
                tokens = cls.ternary_body_write(body, code_string, assignee, tokens, ret=True)
                break
            else:
                tokens += JavaAndCPPProcessor.return_ternary_removal(code_string, ret_tern_expr, child, parser)
        return tokens

    @classmethod
    def extract_ternary_expression(cls, root):
        assi_ten_expr = []
        varde_ten_expr = []
        ret_ten_expr = []
        queue = [root]
        while len(queue) > 0:
            current_node = queue[0]
            queue = queue[1:]
            if str(current_node.type) == 'ternary_expression' and str(
                    current_node.parent.type) == "assignment_expression":
                assi_ten_expr.append(current_node.parent)
            if str(current_node.type) == 'ternary_expression' and str(
                    current_node.parent.type) == "variable_declarator":
                varde_ten_expr.append(current_node.parent.parent)
            if str(current_node.type) == 'ternary_expression' and str(current_node.parent.type) == "return_statement":
                ret_ten_expr.append(current_node.parent)
            for child in current_node.children:
                queue.append(child)
        return assi_ten_expr, varde_ten_expr, ret_ten_expr

    # -----Post increment/decrement removal------
    # -----Post increment/decrement removal------
    @classmethod
    def incre_decre_removal(cls, code_string, parser):
        # code_string = cls.remove_package_and_import(code_string)
        root = parser.parse_code(code_string)
        pre_expr, post_expr = cls.extract_incre_decre_expression(root, code_string)
        
        replacements = []
        if isinstance(code_string, str):
            code_bytes = code_string.encode('utf-8')
        import re 
        # Ensure code_bytes is bytes-like for slicing
        if isinstance(code_bytes, str): code_bytes = code_bytes.encode('utf-8') 
        
        # 1. Pre-Increment: x = ++y -> y+=1; x=y;
        for node in pre_expr:
            try:
                # node is expression_statement
                # children: [assignment_expression, ;]
                assign = node.children[0]
                if assign.type != 'assignment_expression': continue
                
                # assign children: [left, =, right]
                left = assign.children[0]
                right = assign.children[2]
                
                # right is prefix_update (e.g. ++y)
                # children: [++, operand]
                if len(right.children) < 2: continue
                op_node = right.children[0] # ++ or --
                operand = right.children[1]
                
                op_str = code_bytes[op_node.start_byte:op_node.end_byte].decode()
                arith_op = "+=" if "++" in op_str else "-="
                
                left_text = code_bytes[left.start_byte:left.end_byte].decode()
                operand_text = code_bytes[operand.start_byte:operand.end_byte].decode()
                
                new_text = f"{operand_text} {arith_op} 1; {left_text} = {operand_text};"
                replacements.append((node.start_byte, node.end_byte, new_text))
            except:
                pass

        # 2. Post-Increment: x = y++ -> x=y; y+=1;
        for node in post_expr:
            try:
                assign = node.children[0]
                if assign.type != 'assignment_expression': continue
                
                left = assign.children[0]
                right = assign.children[2]
                
                # right is postfix_update (e.g. y++)
                # children: [operand, ++]
                if len(right.children) < 2: continue
                operand = right.children[0]
                op_node = right.children[1]
                
                op_str = code_bytes[op_node.start_byte:op_node.end_byte].decode()
                arith_op = "+=" if "++" in op_str else "-="
                
                left_text = code_bytes[left.start_byte:left.end_byte].decode()
                operand_text = code_bytes[operand.start_byte:operand.end_byte].decode()
                
                new_text = f"{left_text} = {operand_text}; {operand_text} {arith_op} 1;"
                replacements.append((node.start_byte, node.end_byte, new_text))
            except:
                pass
                
        if replacements:
            new_code = cls.apply_replacements(code_string, replacements)
            return parser.parse_code(new_code), new_code, True
            
        return root, code_string, False

    @classmethod
    def pre_incre_decre_removal(cls, code_string, pre_expr, root, parser):
        if isinstance(code_string, str):
            code_string = code_string.encode()
        assert isinstance(root, Node)
        tokens = []
        children = root.children
        if len(children) == 0:
            tokens.append(cls.handle_terminal_node(root, code_string))
        for child in children:
            if child in pre_expr:
                expr = child.children[0]
                assignee = expr.children[0]
                assignee_token = get_tokens(code_string, assignee)[0]
                # check it is increment or decrement
                op = ""
                if str(expr.children[2].children[0].type) == "--":
                    op = "-="
                elif str(expr.children[2].children[0].type) == "++":
                    op = "+="
                assigner = expr.children[2].children[-1]
                assigner_token = get_tokens(code_string, assigner)[0]
                tokens.extend([assigner_token, op, "1", ";", assignee_token, "=", assigner_token, ";"])
                # break
            else:
                tokens += JavaAndCPPProcessor.pre_incre_decre_removal(code_string, pre_expr, child, parser)
        return tokens

    @classmethod
    def post_incre_decre_removal(cls, code_string, post_expr, root, parser):
        if isinstance(code_string, str):
            code_string = code_string.encode()
        assert isinstance(root, Node)
        tokens = []
        children = root.children
        if len(children) == 0:
            tokens.append(cls.handle_terminal_node(root, code_string))
        for child in children:
            if child in post_expr:
                expr = child.children[0]
                assignee = expr.children[0]
                assignee_token = get_tokens(code_string, assignee)[0]
                op = ""
                if str(expr.children[2].children[-1].type) == "--":
                    op = "-="
                elif str(expr.children[2].children[-1].type) == "++":
                    op = "+="
                assigner = expr.children[2].children[0]
                assigner_token = get_tokens(code_string, assigner)[0]
                tokens.extend([assignee_token, "=", assigner_token, ";", assigner_token, op, "1", ";"])
                # break
            else:
                tokens += JavaAndCPPProcessor.post_incre_decre_removal(code_string, post_expr, child, parser)
        return tokens

    @classmethod
    def extract_incre_decre_expression(cls, root, code_string):
        pre_expr = []
        post_expr = []
        queue = [root]
        while len(queue) > 0:
            current_node = queue[0]
            queue = queue[1:]
            if (str(current_node.type) == '++' or str(current_node.type) == "--") and \
                    (str(current_node.parent.type) == "update_expression" or str(
                        current_node.parent.type) == "postfix_unary_expression" or str(
                        current_node.parent.type) == "prefix_unary_expression") and \
                    str(current_node.parent.parent.type) == "assignment_expression":
                nodes = current_node.parent.parent.children
                if len(nodes) == 3 and str(nodes[
                                               0].type) == "identifier":  # this line is to double check whether the
                    # unary operation happens inside an assinemnt expression
                    if str(nodes[2].children[0].type) == "++" or str(nodes[2].children[0].type) == "--":
                        pre_expr.append(current_node.parent.parent.parent)
                    else:
                        post_expr.append(current_node.parent.parent.parent)
            for child in current_node.children:
                queue.append(child)
        return pre_expr, post_expr

    @classmethod
    def handle_terminal_node(cls, root_node, code_string):
        if root_node.type == "comment":
            str_const = ""
        else:
            str_const = code_string[root_node.start_byte:root_node.end_byte].decode("utf-8")
        return str_const

    @classmethod
    def remove_package_and_import(cls, code):
        if isinstance(code, str):
            code = code.encode()
        code = code.decode().split("\n")
        lines = [line.rstrip("\n") for line in code]
        current_code_lines = []
        for line in lines:
            if line.strip().startswith("import") or line.strip().startswith("package") or line.strip().startswith(
                    "#include"):
                # TODO: How to deal with the #if_def kind of code?
                continue
            current_code_lines.append(line)
        code = "\n".join(current_code_lines) if len(current_code_lines) else ""
        return code.encode()

    @classmethod
    def extract_expression(self, root, code):
        expressions = []
        queue = [root]
        while len(queue) > 0:
            current_node = queue[0]
            queue = queue[1:]
            if str(current_node.type) == 'binary_expression':
                children_nodes = current_node.children
                keep = ["<", ">", "<=", ">=", "==", "!="]
                counter = 0
                for w in children_nodes:
                    if str(w.type) in keep:
                        counter = counter + 1
                if counter == 1:
                    expressions.append(current_node)
            for child in current_node.children:
                queue.append(child)
        return expressions

    @classmethod
    def get_tokens_for_opswap(cls, code, root, left_oprd, operator, right_oprd):
        if isinstance(code, str):
            code = code.encode()
        assert isinstance(root, Node)
        tokens = []
        if root.type == "comment":
            return tokens, None
        if "string" in str(root.type):
            return [code[root.start_byte:root.end_byte].decode()], None
        children = root.children
        if len(children) == 0:

            if root.start_byte == operator.start_byte and root.end_byte == operator.end_byte:
                opt = (code[operator.start_byte:operator.end_byte].decode())
                if opt == '<':
                    tokens.append('>')
                elif opt == '>':
                    tokens.append('<')
                elif opt == '>=':
                    tokens.append('<=')
                elif opt == '<=':
                    tokens.append('>=')
                elif opt == '==':
                    tokens.append('==')
                elif opt == '!=':
                    tokens.append('!=')
            else:
                tokens.append(code[root.start_byte:root.end_byte].decode())
        for child in children:
            if child.start_byte == left_oprd.start_byte and child.end_byte == left_oprd.end_byte:
                ts, _ = cls.get_tokens_for_opswap(code, right_oprd, left_oprd, operator, right_oprd)
            elif child.start_byte == right_oprd.start_byte and child.end_byte == right_oprd.end_byte:
                ts, _ = cls.get_tokens_for_opswap(code, left_oprd, left_oprd, operator, right_oprd)
            else:
                ts, _ = cls.get_tokens_for_opswap(code, child, left_oprd, operator, right_oprd)
            tokens += ts
        return tokens, None

    @classmethod
    def operand_swap(cls, code_str, parser):
        code = code_str.encode()
        root = parser.parse_code(code)
        expressions = cls.extract_expression(root, code)
        success = False
        try:
            while not success and len(expressions) > 0:
                selected_exp = np.random.choice(expressions)
                expressions.remove(selected_exp)
                bin_exp = selected_exp
                condition = code[bin_exp.start_byte:bin_exp.end_byte].decode()
                bin_exp = bin_exp.children
                left_oprd = bin_exp[0]
                operator = bin_exp[1]
                right_oprd = bin_exp[2]

                try:
                    code_list = cls.get_tokens_for_opswap(code, root, left_oprd, operator, right_oprd)[0]
                    code_string = ""
                    for w in code_list:
                        code_string = code_string + w + " "
                    code_string = code_string.strip()
                    success = True
                except:
                    success = False
                    continue
        except:
            pass
        if not success:
            code_string = cls.beautify_java_code(get_tokens(code_str, root))
        return code_string, success

    @classmethod
    def extract_if_else(cls, root, code_str, operator_list):
        ext_opt_list = ["&&", "&", "||", "|"]
        expressions = []
        queue = [root]

        not_consider = []
        while len(queue) > 0:
            current_node = queue[0]
            queue = queue[1:]
            if str(current_node.type) == 'if_statement':
                clause = code_str[current_node.start_byte:current_node.end_byte].decode()
                des = (current_node.children)[1]
                cond = code_str[des.start_byte:des.end_byte].decode()
                stack = [des]
                nodes = []
                while len(stack) > 0:
                    root1 = stack.pop()
                    if len(root1.children) == 0:
                        nodes.append(root1)

                    for child in root1.children:
                        stack.append(child)
                nodes.reverse()
                counter = 0
                extra_counter = 0
                for w in nodes:
                    if str(w.type) in operator_list:
                        counter = counter + 1
                    if str(w.type) in ext_opt_list:
                        extra_counter = extra_counter + 1
                if not (counter == 1 and extra_counter == 0):
                    continue
                children_nodes = current_node.children
                flagx = 0
                flagy = 0
                for w in children_nodes:
                    if str(w.type) == "else":
                        flagx = 1
                    if str(w.type) == "if_statement":
                        not_consider.append(w)
                        flagy = 1
                if flagx == 1 and flagy == 0:
                    expressions.append([current_node, des])
            for child in current_node.children:
                if child not in not_consider:
                    queue.append(child)

        return expressions

    @classmethod
    def get_tokens_for_blockswap(cls, code, root, first_block, opt_node, second_block, flagx, flagy):
        if isinstance(code, str):
            code = code.encode()
        assert isinstance(root, Node)
        tokens = []
        
        if root.type == "comment":
            return tokens, None
        if "string" in str(root.type):
            return [code[root.start_byte:root.end_byte].decode()], None
        children = root.children
        if len(children) == 0:
            if root.start_byte == opt_node.start_byte and root.end_byte == opt_node.end_byte:
                op = code[root.start_byte:root.end_byte].decode()
                if op == "<":
                    tokens.append(">=")
                elif op == ">":
                    tokens.append('<=')
                elif op == ">=":
                    tokens.append('<')
                elif op == "<=":
                    tokens.append('>')
                elif op == "!=":
                    tokens.append('==')
                elif op == "==":
                    tokens.append('!=')
            else:
                tokens.append(code[root.start_byte:root.end_byte].decode())
        for child in children:
            child_type = str(child.type)
            if child.start_byte == first_block.start_byte and child.end_byte == first_block.end_byte and flagx == 0 \
                    and str(
                child.type) == str(first_block.type):
                flagx = 1
                ts, _ = cls.get_tokens_for_blockswap(code, second_block, first_block, opt_node, second_block, flagx,
                                                     flagy)

            elif child.start_byte == second_block.start_byte and child.end_byte == second_block.end_byte and flagy == \
                    0 and str(
                child.type) == str(second_block.type):
                flagy = 1
                ts, _ = cls.get_tokens_for_blockswap(code, first_block, first_block, opt_node, second_block, flagx,
                                                     flagy)

            else:
                ts, _ = cls.get_tokens_for_blockswap(code, child, first_block, opt_node, second_block, flagx, flagy)
            tokens += ts

        return tokens, None

    @classmethod
    def block_swap_java(cls, code_str, parser):
        """
        Swap if/else blocks for Java. 
        Reuses the robust C/C++ implementation as the Tree-sitter structure for if-else is compatible.
        preserves formatting.
        """
        return cls.block_swap_c(code_str, parser)

    @classmethod
    def block_swap_c(cls, code_str, parser):
        """
        Swap if/else blocks while negating the comparison operator to preserve semantics.
        Supports both C (typically uses else_clause) and C++ (typically flat if structure) ASTs.
        """
        if isinstance(code_str, bytes):
            code_bytes = code_str
            code_str = code_str.decode()
        else:
            code_bytes = code_str.encode()

        root = parser.parse_code(code_bytes)
        operator_map = {
            "<": ">=",
            ">": "<=",
            "<=": ">",
            ">=": "<",
            "==": "!=",
            "!=": "==",
        }

        def find_swappable_if(node):
            queue = [node]
            while queue:
                cur = queue.pop(0)
                if cur.type == "if_statement":
                    conseq = cur.child_by_field_name("consequence")
                    alt = cur.child_by_field_name("alternative")
                    cond = cur.child_by_field_name("condition")
                    
                    # Manual scan if fields are missing (common in some grammar versions)
                    children = cur.children
                    
                    # Find condition if missing
                    if not cond:
                        for child in children:
                            if child.type in ["condition_clause", "parenthesized_expression"]:
                                cond = child
                                break
                                
                    # Find consequence if missing (usually after condition)
                    if not conseq and cond:
                        # Consequence is typically the next named sibling or block
                        found_cond = False
                        for child in children:
                            if child == cond:
                                found_cond = True
                                continue
                            if found_cond and child.type == "compound_statement":
                                conseq = child
                                break

                    # Find alternative (else block)
                    else_block = None
                    if alt:
                        if alt.type == "else_clause":
                            # Extract actual block from else_clause
                            for grandchild in alt.children:
                                if grandchild.type == "compound_statement" or grandchild.type == "if_statement" or grandchild.type == "block":
                                    else_block = grandchild
                                    break
                            if not else_block and len(alt.children) > 0:
                                else_block = alt.children[-1]
                        else:
                            else_block = alt
                    else:
                        # Manual scan for 'else' keyword
                        for i, child in enumerate(children):
                            if child.type == "else" or child.type == "else_clause":
                                if child.type == "else_clause":
                                    # Recursive check for else_clause
                                    for grandchild in child.children:
                                        if grandchild.type in ["compound_statement", "block", "if_statement"]:
                                            else_block = grandchild
                                    if not else_block: else_block = child.children[-1]
                                elif i + 1 < len(children):
                                    else_block = children[i+1]
                                break
                    
                    if cond and conseq and else_block:
                        return cur, cond, conseq, else_block

                queue.extend(cur.children)
            return None, None, None, None

        success = False
        if_node, cond_node, conseq_node, else_block = find_swappable_if(root)
        
        if if_node:
            try:
                # Extract binary expression
                cond_expr = cond_node
                if cond_expr.type in ["condition_clause", "parenthesized_expression"] and len(cond_expr.children) >= 2:
                     # Usually child[1] is the expression inside ( expr )
                     cond_expr = cond_expr.children[1]
                
                # Check for binary expression
                if cond_expr.type == "binary_expression" and len(cond_expr.children) >= 3:
                    left, op_node, right = cond_expr.children[0], cond_expr.children[1], cond_expr.children[2]
                    op_text = code_bytes[op_node.start_byte:op_node.end_byte].decode()
                    
                    if op_text in operator_map:
                        neg_cond = (
                            code_bytes[left.start_byte:left.end_byte]
                            + b" " + operator_map[op_text].encode() + b" "
                            + code_bytes[right.start_byte:right.end_byte]
                        )
                        new_condition = b"(" + neg_cond + b")"

                        then_block = code_bytes[conseq_node.start_byte:conseq_node.end_byte]
                        else_content = code_bytes[else_block.start_byte:else_block.end_byte]

                        # Reconstruct the if statement
                        # Note: We need to be careful about not duplicating 'else' keyword if it's already there
                        # But simpler approach: verify structure is standard `if (...) { } else { }`
                        
                        # Safe reconstruction mapping:
                        # PRE_COND + NEW_COND + PRE_CONSEQ + ELSE_CONTENT + PRE_ELSE + THEN_CONTENT
                        # This assumes standard layout. safer to replace specific ranges.
                        
                        # Actually, typically structure: if (COND) THEN else ELSE
                        # We want: if (!COND) ELSE else THEN
                        
                        # Find 'else' keyword position to anchor the swap
                        else_keyword = None
                        for child in if_node.children:
                            if child.type == "else" or (child.type == "else_clause" and child.start_byte < else_block.start_byte):
                                else_keyword = child
                                break
                        
                        if else_keyword:
                             # Construct new string
                             # Part 1: 'if ' ... '('
                             part1 = code_bytes[if_node.start_byte:cond_node.start_byte]
                             # Part 2: new condition
                             # Part 3: ')' ... '{' (before then block) - wait, cond_node includes parens usually? 
                             # If cond_node is condition_clause, it includes parens.
                             
                             if cond_node.type == "condition_clause":
                                 # replace content inside parens
                                 part2 = new_condition # built with parens above
                             else:
                                 # cond_node might be just parenthesized_expression
                                 part2 = new_condition

                             # Space between cond and consq
                             part3 = code_bytes[cond_node.end_byte:conseq_node.start_byte]
                             
                             # New Then Block (Old Else)
                             part4 = else_content
                             
                             # Space between Then and Else keyword (rare/empty)
                             part5 = code_bytes[conseq_node.end_byte:else_keyword.start_byte]
                             
                             # Else Keyword + space
                             # If else_clause wraps, else_keyword start might strictly precede else_block start
                             # We use else_keyword from AST
                             part6 = code_bytes[else_keyword.start_byte:else_block.start_byte]
                             if else_keyword.type == "else_clause":
                                 # We need the 'else' text from within else_clause?
                                 # Simplified: just use " else " literal if structure is messy?
                                 # No, preserve comments/spaces if possible.
                                 pass
                             
                             # New Else Block (Old Then)
                             part7 = then_block
                             
                             new_if = part1 + part2 + part3 + part4 + part5 + part6 + part7
                             
                             code_bytes = (
                                code_bytes[:if_node.start_byte]
                                + new_if
                                + code_bytes[if_node.end_byte:]
                             )
                             success = True

            except Exception as e:
                # print(f"DEBUG: Error in block_swap_c: {e}")
                success = False

        if success:
            code_string = code_bytes.decode()
        else:
            code_string = cls.beautify_java_code(get_tokens(code_str, root))
        return code_string, success
