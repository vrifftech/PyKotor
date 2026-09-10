"""PLY validates lexer/parser rules using inspect.getsourcelines().

Keep these grammar modules as source alongside their normal PYZ bytecode.
Without source, an application can launch successfully but fail to compile NSS.
"""
module_collection_mode = "pyz+py"
