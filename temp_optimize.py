import re

def optimize_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex to match: <series_expr>.rolling(<window>).<func>()
    # where func is mean, std, min, max, sum, var
    # Example: df['close'].rolling(20).mean() -> self._rolling(df['close'], 20, 'mean')
    
    # We need to be careful with things like: df['tr_14'].rolling(14).mean()
    # It might match properly with: ([a-zA-Z0-9_\[\]\']+) depending on brackets. Let's use a simpler approach.
    
    pattern = re.compile(r'([a-zA-Z0-9_\'\"\[\]]+)\.rolling\((\d+|[a-zA-Z0-9_]+)\)\.(mean|std|min|max|sum|var)\(\)')
    
    def replacer(match):
        series = match.group(1)
        window = match.group(2)
        func = match.group(3)
        return f"self._rolling({series}, {window}, '{func}')"

    new_content, count = pattern.subn(replacer, content)
    
    # Replace the df['close'].replace(0, np.nan) with self._close_safe
    new_content = new_content.replace("df['close'].replace(0, np.nan)", "self._close_safe")
    # Some might use "self.df['close'].replace(0, np.nan)", which is fine.
    
    new_content = new_content.replace("df['volume'].replace(0, np.nan)", "self._volume_safe")
    new_content = new_content.replace("df['open'].replace(0, np.nan)", "self._open_safe")
    new_content = new_content.replace("df['low'].replace(0, np.nan)", "self._low_safe")
    
    hl_str1 = "(df['high'] - df['low']).replace(0, np.nan)"
    hl_str2 = "(self.df['high'] - self.df['low']).replace(0, np.nan)"
    new_content = new_content.replace(hl_str1, "self._hl_range_safe")
    new_content = new_content.replace(hl_str2, "self._hl_range_safe")

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)
        
    print(f"Replaced {count} rolling occurrences!")

optimize_file('data-build/feature_v2.py')
