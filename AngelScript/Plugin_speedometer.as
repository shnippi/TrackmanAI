#name "Speed O Meter"
#author "MosKi"
#category "In Game"

CGameUILayer@ speed_o_meter_layer = null; //This will be our main layer (speed o meter)

CGameUILayer@ Inject_Ingame_Layer(wstring manialink, string id) //Func used to create a manialink layer ingame
{
	auto PlrScript = cast<CTrackManiaNetwork>(GetApp().Network);
	if(PlrScript is null)
		return null;

	auto UIMgr = cast<CGameManiaAppPlayground>(PlrScript.ClientManiaAppPlayground); //This is ClientSide ManiaApp 
		if(UIMgr is null)
		return null;


	auto clientUI = cast<CGamePlaygroundUIConfig>(UIMgr.ClientUI); //We access ClientSide UI class
	if(clientUI is null)
		return null;

	for(uint i = 0; i < clientUI.UILayers.Length; ++i) //This is used to check if we haven't already a layer with the same id, to avoir doubles
	{
		auto layer = cast<CGameUILayer>(clientUI.UILayers[i]);
		if(layer.AttachId == id)
			UIMgr.UILayerDestroy(layer);
	}


	auto injected_layer = UIMgr.UILayerCreate(); //This create a ML Layer in client UI
	injected_layer.AttachId = id; //We'll use AttachId as a reference to get organized in which UI is what
	injected_layer.ManialinkPage = manialink; //This is where the manialink code is
	clientUI.UILayers.Add(injected_layer); // We add the UI Layer to player's UI

	return injected_layer;// The function return the layer pointer to easily modify stuff in it
}

void Main()
{

	wstring ML_page = """ <label pos="145 -81" z-index="0" size="20 16" text="1" style="TextValueMediumSm" valign="center2" halign="center" id="speed"/> """;
	//This is the label manialink that'll be used to create the speedometer
	
	while(true) //This while true is used to create UI each time we change of playground
	{
		yield();
		@speed_o_meter_layer = null; 

		while(speed_o_meter_layer is null) //Inject_Ingame_Layer will return null as long as it can't create a Layer, we wait for layer to be created to continue
		{
			@speed_o_meter_layer = Inject_Ingame_Layer(ML_page, "speedometer");
			yield();
		}

		auto ML_localpage = cast<CGameManialinkPage>(speed_o_meter_layer.LocalPage);//We load Manialink page to function like "GetFirstChild"
		if(ML_localpage is null)
			continue;
		auto Script_Handler = cast<CGameManialinkScriptHandler>(ML_localpage.ScriptHandler);//We load ScriptHandler to get pending events for example


		while(cast<CTrackManiaNetwork>(GetApp().Network).ClientManiaAppPlayground !is null) //This will replace <scripts> parts, it'll be active as long as the UI can be loaded
		{
			
			auto currentPlayground = cast<CSmArenaClient>(GetApp().CurrentPlayground);
			if(currentPlayground is null)
				{yield();continue;}//I don't yield at the begining or pending event will be cleared

			auto players = currentPlayground.GameTerminals;
			if(players.Length <= 0)
				{yield();continue;}

			auto game_terminal = cast<CGameTerminal>(players[0]);
			if(game_terminal is null)
				{yield();continue;}

			auto game_player = cast<CSmPlayer>(game_terminal.GUIPlayer);
			if(game_player is null)
				{yield();continue;}

			auto plr_api = cast<CSmScriptPlayer>(game_player.ScriptAPI); //All those leads to playerScriptApi to retrieve player speed
			if(plr_api is null)
				{yield();continue;}

			auto speed_label = cast<CGameManialinkLabel>(ML_localpage.GetFirstChild("speed"));//We retrieve the Label classes related to our manialink
			if(speed_label is null)
				{yield();continue;}

			speed_label.Value = "" + int(plr_api.Speed*3.6); //We change constantly update the text of it with the player speed in km/h
			

			yield();
		}
	}
}